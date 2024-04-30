import torch

from fastapi import FastAPI, Request
from sse_starlette.sse import EventSourceResponse
from fastapi.responses import StreamingResponse
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from pydantic import BaseModel
from threading import Thread
import uvicorn
import torch
import json
import sys

from typing import List, Dict


class Message(BaseModel):
     role: str
     content: str

class CompletionRequest(BaseModel):
    model: str
    messages: List[Message] = None
    prompt: str = None
    max_tokens: int = 8192
    n: int = 1
    stop: List[str] = None
    temperature: float = 0.2
    top_p: float = 0.8
    stream: bool = False
    do_sample: bool = True


app = FastAPI()

# SafeTensorsモデルのパスを指定
model_path = "CohereForAI/c4ai-command-r-v01"
model_path = "google/gemma-7b-it"
model_path = "codellama/CodeLlama-7b-Instruct-hf"
model_path = "microsoft/Phi-3-mini-128k-instruct"
model_path = sys.argv[1]

model_name = model_path.split('/')[-1]

#from transformers import BitsAndBytesConfig
#quantization_config_4bit = BitsAndBytesConfig(load_in_4bit=True)
#quantization_config_8bit = BitsAndBytesConfig(load_in_8bit=True)

# モデルとトークナイザーのロード
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
#model = AutoModelForCausalLM.from_pretrained(model_path, load_in_4bit=True, device_map="auto")
#model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.float16)
#model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, quantization_config=quantization_config_8bit, low_cpu_mem_usage=True, attn_implementation="flash_attention_2")
tokenizer = AutoTokenizer.from_pretrained(model_path)

@app.post("/v1/{engine}/completions")
async def generate_text(request: Request,engine: str, completion_request: CompletionRequest):

    # SSE用ストリーム作成
    streamer = TextIteratorStreamer(
            tokenizer,    
            skip_prompt=False, # 入力文(ユーザーのプロンプトなど)を出力するかどうか
            skip_special_tokens=True, # その他のデコード時のオプションもここで渡す
    )

    # 入力
    # inputs = tokenizer(completion_request.prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
    chat = []
    system_message = ''
    if completion_request.messages:
        for message in completion_request.messages:
            if message.role == "system":
                # gemmaではシステムプロンプトはないのでまとめてユーザープロンプトに混ぜる。
                system_message += message.content
            else:
                chat.append({"role": message.role, "content": system_message+'\n\n'+message.content})
        prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    elif completion_request.prompt:
        prompt = completion_request.prompt
    else:
        raise HTTPException(status_code=400, detail="message or prompt required.")

    # print('PROMPT=', type(prompt), prompt)
    inputs = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
    # print(inputs)
    inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=True)
    prompt_tokens = inputs['input_ids'].size()[1] - 1
    # print(inputs)
    # attention_mask を生成
    attention_mask = inputs['attention_mask'] if 'attention_mask' in inputs else None
    # attention_mask = inputs.get('attention_mask', None)

    # 推論用引数を設定
    generation_kwargs = dict(
            # input_ids = inputs.to(model.device), 
            input_ids = inputs['input_ids'].to(model.device),
            attention_mask = attention_mask,
            # max_length = completion_request.max_tokens,
            max_new_tokens = completion_request.max_tokens,
            temperature = completion_request.temperature,
            top_p = completion_request.top_p,
            num_return_sequences = completion_request.n,
            # eos_token_id = tokenizer.eos_token_id if completion_request.stop is None else tokenizer.encode(completion_request.stop)[0],
            eos_token_id = tokenizer.eos_token_id if completion_request.stop is None else tokenizer.encode(completion_request.stop[0])[0],
            streamer = streamer if completion_request.stream else None,
            do_sample = completion_request.do_sample,
    )

    if completion_request.stream:
        # ストリームで返す
        pass
    else:
        # バッチ処理で返す
        # print(prompt_tokens)
        # print(prompt_tokens[1])
        outputs = model.generate(**generation_kwargs)
        # print(tokenizer.decode(outputs[0], skip_special_tokens=True))
        decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        total_tokens = len(outputs[0])
        # print(inputs)
        # print(outputs[0])
        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": total_tokens - prompt_tokens,
            "total_tokens": total_tokens
        }
        return {"model": model_name, "usage": usage, "choices":[{"index":0,"delta":{"role":"assistant","content": decoded_output[len(prompt):] }}]}

    # 別スレッドで推論を実行
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    async def event_generator():
        generated_text = ""
        token_count = 0
        # streamer に別スレッドでの推論結果が流れ込んでくる
        for decoded_output in streamer:
            # ログ用に全部結合した結果を保持しておく
            generated_text += decoded_output
            token_count += 1
            # ストリームで返却
            yield {"data": json.dumps({"model": model_name, "choices":[{"index":0,"delta":{"role":"assistant","content": decoded_output }}]})}
        # ストリーム終了
        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": token_count,
            "total_tokens": prompt_tokens + token_count
        }
        yield {"data": json.dumps({"model": model_name, "usage": usage, "choices":[{"index":0,"delta":{}}]})}
        yield "[DONE]"

    # SSEで返却
    return EventSourceResponse(event_generator())

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=3000)
