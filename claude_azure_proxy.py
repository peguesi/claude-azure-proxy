#!/usr/bin/env python3
"""
Minimal Anthropic /v1/messages -> Azure chat completions proxy.
Configurable via env:
- AZURE_OPENAI_KEY (required)
- AZURE_OPENAI_BASE (e.g. https://your-resource.openai.azure.com)
- AZURE_API_VERSION (default: 2024-10-01-preview)
"""
import os
import json
import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse
import uvicorn

app = FastAPI()

MODEL_MAP = {
    "claude-3-5-sonnet-20241022": "gpt-5.1",
    "claude-sonnet-4-5-20250929": "gpt-5.1",
    "claude-3-opus-20240229": "gpt-5.1",
    "claude-opus-4-5-20251101": "gpt-5.1",
    "claude-3-5-haiku-20241022": "gpt-5-nano",
    "claude-haiku-4-5-20251001": "gpt-5-nano",
}

AZURE_BASE = os.environ.get("AZURE_OPENAI_BASE")
AZURE_API_VERSION = os.environ.get("AZURE_API_VERSION", "2024-10-01-preview")
AZURE_KEY = os.environ.get("AZURE_OPENAI_KEY")

if not AZURE_KEY or not AZURE_BASE:
    raise ValueError("AZURE_OPENAI_KEY and AZURE_OPENAI_BASE environment variables are required")


def anthropic_to_azure(payload: dict) -> dict:
    messages = []
    if "system" in payload:
        messages.append({"role": "system", "content": payload["system"]})
    for msg in payload.get("messages", []):
        role = msg["role"]
        content = msg["content"]
        if isinstance(content, str):
            messages.append({"role": role, "content": content})
        elif isinstance(content, list):
            text_parts = [block.get("text", "") for block in content if block.get("type") == "text"]
            messages.append({"role": role, "content": "\n".join(text_parts)})
    azure_payload = {
        "messages": messages,
        "temperature": payload.get("temperature", 1.0),
        "stream": payload.get("stream", False),
    }
    azure_payload["max_completion_tokens"] = payload.get("max_tokens", 4096)
    return azure_payload


def azure_to_anthropic_chunk(chunk: dict, model: str):
    if not chunk.get("choices"):
        return None
    choice = chunk["choices"][0]
    delta = choice.get("delta", {})
    if choice.get("index") == 0 and not delta.get("content"):
        return {
            "type": "message_start",
            "message": {
                "id": chunk.get("id", "msg_001"),
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": model,
            }
        }
    if delta.get("content"):
        return {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": delta["content"]},
        }
    if choice.get("finish_reason"):
        return {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 0},
        }
    return None


@app.post("/v1/messages")
async def messages(request: Request):
    payload = await request.json()
    model = payload.get("model", "claude-3-5-sonnet-20241022")
    deployment = MODEL_MAP.get(model, "gpt-5.1")
    azure_payload = anthropic_to_azure(payload)
    stream = azure_payload.pop("stream", False)

    url = f"{AZURE_BASE}/openai/deployments/{deployment}/chat/completions"
    params = {"api-version": AZURE_API_VERSION}
    headers = {"api-key": AZURE_KEY, "Content-Type": "application/json"}

    async with httpx.AsyncClient(timeout=120.0) as client:
        if stream:
            azure_payload["stream"] = True

            async def stream_anthropic():
                async with client.stream("POST", url, params=params, headers=headers, json=azure_payload) as response:
                    async for line in response.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        data = line[6:]
                        if data == "[DONE]":
                            continue
                        try:
                            chunk = json.loads(data)
                            anthropic_chunk = azure_to_anthropic_chunk(chunk, model)
                            if anthropic_chunk:
                                yield f"event: {anthropic_chunk['type']}\n"
                                yield f"data: {json.dumps(anthropic_chunk)}\n\n"
                        except json.JSONDecodeError:
                            continue
                    yield 'event: message_stop\ndata: {}\n\n'

            return StreamingResponse(stream_anthropic(), media_type="text/event-stream")
        else:
            response = await client.post(url, params=params, headers=headers, json=azure_payload)
            if response.status_code != 200:
                return Response(
                    content=json.dumps({
                        "error": {
                            "message": response.text,
                            "status_code": response.status_code,
                            "azure_url": url,
                            "deployment": deployment,
                        }
                    }),
                    status_code=500,
                    media_type="application/json",
                )

            azure_response = response.json()
            if not azure_response.get("choices"):
                return Response(
                    content=json.dumps({"error": {"message": "No choices in Azure response", "body": azure_response}}),
                    status_code=500,
                    media_type="application/json",
                )

            content_text = azure_response["choices"][0]["message"]["content"]
            anthropic_response = {
                "id": azure_response.get("id", "msg_001"),
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": content_text}],
                "model": model,
                "stop_reason": "end_turn",
                "usage": {
                    "input_tokens": azure_response.get("usage", {}).get("prompt_tokens", 0),
                    "output_tokens": azure_response.get("usage", {}).get("completion_tokens", 0),
                },
            }
            return Response(content=json.dumps(anthropic_response), media_type="application/json")


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    print("🚀 Claude → Azure OpenAI Proxy")
    print("   Listening on http://127.0.0.1:8100")
    print(f"   Azure base: {AZURE_BASE}")
    print(f"   Model mappings: {len(MODEL_MAP)} Claude models")
    print()
    uvicorn.run(app, host="0.0.0.0", port=8100, log_level="info")
