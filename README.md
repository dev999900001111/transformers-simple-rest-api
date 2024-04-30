Here's the updated README with the `usage` section including the `microsoft/Phi-3-mini-128k-instruct` model:

# transformers-rest-api

This program is a simple Python script that allows you to use the `transformers` library through a RESTful API. The API interface is designed to be similar to the OpenAI API format.

## Features

- Provides a RESTful API using FastAPI
- Request and response formats similar to the OpenAI API
- Supports streaming for text generation
- Supports multiple models (specified by model path as an argument)
- Fast inference using multi-threading

## Dependencies

- Python 3.7 or higher
- FastAPI
- transformers
- torch
- uvicorn

## Usage

1. Install the dependencies.

```bash
pip install fastapi transformers torch uvicorn
```

2. Run the program, specifying the model path as an argument. For example, to use the `microsoft/Phi-3-mini-128k-instruct` model:

```bash
python app.py microsoft/Phi-3-mini-128k-instruct
```

3. Send a POST request to the API endpoint to generate text.

```bash
curl -X POST "http://localhost:3000/v1/completions" \
     -H "Content-Type: application/json" \
     -d '{
           "model": "Phi-3-mini-128k-instruct",
           "messages": [{"role": "user", "content": "Hello, how are you?"}],
           "max_tokens": 50,
           "temperature": 0.7,
           "stream": true
         }'
```

## API Reference

### POST /v1/{engine}/completions

Generates text.

#### Request Body

| Parameter   | Type    | Description                                        |
|-------------|---------|---------------------------------------------------|
| model       | string  | Name of the model to use(not implemented yet)      |
| messages    | array   | Array of message objects (system and user messages)|
| prompt      | string  | Prompt text (if messages are not specified)        |
| max_tokens  | integer | Maximum number of tokens to generate               |
| n           | integer | Number of responses to generate                    |
| stop        | array   | List of tokens to stop generation                  |
| temperature | float   | Temperature for randomness (range: 0 to 2)         |
| top_p       | float   | Probability for top_p sampling                     |
| stream      | boolean | Whether to enable streaming output                 |
| do_sample   | boolean | Whether to enable sampling output                  |

#### Response

If streaming is disabled, a JSON object containing the generated text is returned.
If streaming is enabled, the text is returned in Server-Sent Events (SSE) format.

## License

This project is released under the MIT License.