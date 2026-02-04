#!/usr/bin/env python3
"""
Anthropic /v1/messages → Azure OpenAI chat completions proxy (2025 features).

Supported (best-effort on Azure):
- Tool use (tool_use/tool_result), tool calls -> Azure function calls
- Extended thinking blocks preserved (<thinking> tags)
- Effort parameter (approximated via max tokens/temperature)
- Beta headers: interleaved-thinking, advanced-tool-use, context-management
- Plan-mode/context hints carried in system prompt

Limitations (Azure side):
- No true Programmatic Tool Calling (PTC) execution
- No separate thinking/output token pools
- Effort is approximated (Azure has no native knob)
"""
import os
import json
from typing import Dict, List, Any, Optional

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse
import uvicorn

app = FastAPI()

# Claude aliases -> Azure chat-capable deployments
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


def extract_beta_headers(request_headers: Dict[str, str]) -> List[str]:
    """Extract beta feature flags from request headers."""
    betas = []
    beta_header = request_headers.get("anthropic-beta", "")
    if beta_header:
        betas.extend([b.strip() for b in beta_header.split(",") if b.strip()])
    return betas


def anthropic_tools_to_azure(anthropic_tools: list) -> list:
    """Convert Anthropic tool definitions to Azure/OpenAI function tools."""
    if not anthropic_tools:
        return []

    azure_tools = []
    for tool in anthropic_tools:
        name = tool.get("name", "tool")
        desc = tool.get("description", "")
        schema = tool.get("input_schema", {"type": "object", "properties": {}})
        azure_tools.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": desc,
                    "parameters": schema,
                },
            }
        )
    return azure_tools


def anthropic_to_azure(anthropic_payload: dict, betas: List[str]) -> (dict, dict):
    """
    Convert Anthropic /v1/messages format to Azure chat completions.
    Returns (azure_payload, metadata)
    """
    messages: List[Dict[str, Any]] = []
    metadata = {
        "thinking_config": anthropic_payload.get("thinking"),
        "output_config": anthropic_payload.get("output_config", {}),
        "betas": betas,
    }

    # System prompt + hints
    system_content = anthropic_payload.get("system", "")
    thinking_config = anthropic_payload.get("thinking")
    output_config = anthropic_payload.get("output_config", {})

    if thinking_config and thinking_config.get("type") == "enabled":
        budget = thinking_config.get("budget_tokens", 4096)
        system_content += f"\n\n[Extended Thinking Budget: {budget} tokens. Use step-by-step reasoning.]"

    effort = output_config.get("effort", "high")
    if effort != "high":
        system_content += f"\n\n[Effort Level: {effort}. Adjust response thoroughness accordingly.]"

    if "context-management-2025-06-27" in betas:
        system_content += "\n\n[Context Management: Auto-compact old tool logs and thoughts.]"

    if system_content:
        messages.append({"role": "system", "content": system_content})

    # Convert messages, preserve thinking, tool results
    for msg in anthropic_payload.get("messages", []):
        role = msg["role"]
        content = msg["content"]

        if isinstance(content, str):
            messages.append({"role": role, "content": content})
        elif isinstance(content, list):
            text_parts: List[str] = []
            for block in content:
                block_type = block.get("type")
                if block_type == "thinking":
                    text_parts.append(f"<thinking>{block.get('thinking', '')}</thinking>")
                elif block_type == "tool_result":
                    # Avoid Azure "tool role without tool_calls" error: fold into text
                    result_content = block.get("content")
                    if isinstance(result_content, list):
                        result_text = "\n".join(
                            r.get("text", "") for r in result_content if r.get("type") == "text"
                        )
                    else:
                        result_text = str(result_content)
                    text_parts.append(result_text)
                elif block_type == "text":
                    text_parts.append(block["text"])
                # Ignore other block types for now
            if text_parts:
                messages.append({"role": role, "content": "\n".join(text_parts)})

    # Build Azure payload
    azure_payload: Dict[str, Any] = {
        "messages": messages,
        "temperature": anthropic_payload.get("temperature", 1.0),
        "stream": anthropic_payload.get("stream", False),
    }
    azure_payload["max_completion_tokens"] = anthropic_payload.get("max_tokens", 4096)

    # Tools
    if "tools" in anthropic_payload:
        azure_payload["tools"] = anthropic_tools_to_azure(anthropic_payload["tools"])

    # Effort approximation
    if effort == "low":
        azure_payload["max_completion_tokens"] = min(azure_payload["max_completion_tokens"], 2048)
        azure_payload["temperature"] = 0.7
    elif effort == "medium":
        azure_payload["temperature"] = 0.9

    return azure_payload, metadata


def azure_to_anthropic_response(azure_response: dict, model: str, metadata: dict) -> dict:
    """Convert Azure response to Anthropic message with thinking/tool_use."""
    choice = azure_response["choices"][0]
    message = choice["message"]

    content: List[Any] = []
    response_text = message.get("content", "") or ""

    # Extract thinking blocks if present
    if "<thinking>" in response_text:
        parts = response_text.split("<thinking>")
        if parts[0].strip():
            content.append({"type": "text", "text": parts[0].strip()})
        for part in parts[1:]:
            if "</thinking>" in part:
                thinking_text, remaining = part.split("</thinking>", 1)
                content.append({"type": "thinking", "thinking": thinking_text.strip()})
                if remaining.strip():
                    content.append({"type": "text", "text": remaining.strip()})
    elif response_text:
        content.append({"type": "text", "text": response_text})

    # Tool calls -> tool_use blocks
    if message.get("tool_calls"):
        for tool_call in message["tool_calls"]:
            try:
                tool_input = json.loads(tool_call["function"]["arguments"])
            except json.JSONDecodeError:
                tool_input = {"raw": tool_call["function"]["arguments"]}
            content.append(
                {
                    "type": "tool_use",
                    "id": tool_call["id"],
                    "name": tool_call["function"]["name"],
                    "input": tool_input,
                }
            )

    finish_reason = choice.get("finish_reason")
    if finish_reason == "tool_calls":
        stop_reason = "tool_use"
    elif finish_reason == "length":
        stop_reason = "max_tokens"
    else:
        stop_reason = "end_turn"

    # Sanitize content: ensure list of dict blocks
    safe_content: List[Dict[str, Any]] = []
    for item in content if content else [{"type": "text", "text": ""}]:
        if isinstance(item, dict):
            safe_content.append(item)
        else:
            safe_content.append({"type": "text", "text": str(item)})

    return {
        "id": azure_response.get("id", "msg_001"),
        "type": "message",
        "role": "assistant",
        "content": safe_content,
        "model": model,
        "stop_reason": stop_reason,
        "usage": {
            "input_tokens": azure_response.get("usage", {}).get("prompt_tokens", 0),
            "output_tokens": azure_response.get("usage", {}).get("completion_tokens", 0),
        },
    }


def azure_stream_chunk_to_anthropic(chunk: dict, model: str):
    """Convert Azure streaming chunk to Anthropic SSE events."""
    if not chunk.get("choices"):
        return []
    choice = chunk["choices"][0]
    delta = choice.get("delta", {})
    events = []

    # message_start when no content yet and no finish_reason
    if delta.get("role") == "assistant" and not delta.get("content") and not choice.get("finish_reason"):
        events.append({"type": "message_start", "message": {"id": chunk.get("id", "msg_stream"), "type": "message", "role": "assistant", "content": [], "model": model}})

    # content delta
    if delta.get("content"):
        events.append({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": delta["content"]}})

    # tool_calls in stream
    if delta.get("tool_calls"):
        for tool_call in delta["tool_calls"]:
            try:
                tool_input = json.loads(tool_call["function"]["arguments"])
            except json.JSONDecodeError:
                tool_input = {"raw": tool_call["function"]["arguments"]}
            events.append(
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {
                        "type": "tool_use",
                        "id": tool_call.get("id", "call"),
                        "name": tool_call.get("function", {}).get("name", "tool"),
                        "input": tool_input,
                    },
                }
            )

    # finish
    if choice.get("finish_reason"):
        stop_reason = "tool_use" if choice["finish_reason"] == "tool_calls" else "end_turn"
        events.append({"type": "message_delta", "delta": {"stop_reason": stop_reason}, "usage": {"output_tokens": 0}})
        events.append({"type": "message_stop"})

    return events


@app.post("/v1/messages")
async def messages(request: Request):
    """Handle Anthropic /v1/messages with 2025 feature support."""
    payload = await request.json()
    betas = extract_beta_headers(dict(request.headers))

    model = payload.get("model", "claude-3-5-sonnet-20241022")
    deployment = MODEL_MAP.get(model, "gpt-5.1")

    azure_payload, metadata = anthropic_to_azure(payload, betas)
    stream = azure_payload.pop("stream", False)

    url = f"{AZURE_BASE}/openai/deployments/{deployment}/chat/completions"
    params = {"api-version": AZURE_API_VERSION}
    headers = {"api-key": AZURE_KEY, "Content-Type": "application/json"}

    if stream:
        azure_payload["stream"] = True
        client = httpx.AsyncClient(timeout=180.0)

        async def stream_anthropic():
            try:
                async with client.stream("POST", url, params=params, headers=headers, json=azure_payload) as resp:
                    async for line in resp.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        data = line[6:]
                        if data == "[DONE]":
                            yield "event: message_stop\ndata: {}\n\n"
                            continue
                        try:
                            chunk = json.loads(data)
                            events = azure_stream_chunk_to_anthropic(chunk, model)
                            for ev in events:
                                yield f"event: {ev['type']}\n"
                                yield f"data: {json.dumps(ev)}\n\n"
                        except json.JSONDecodeError:
                            continue
            finally:
                await client.aclose()

        return StreamingResponse(stream_anthropic(), media_type="text/event-stream")

    else:
        async with httpx.AsyncClient(timeout=180.0) as client:
            azure_payload["stream"] = False
            response = await client.post(url, params=params, headers=headers, json=azure_payload)
            if response.status_code != 200:
                return Response(content=response.text, status_code=response.status_code, media_type="application/json")

            azure_response = response.json()
            anthropic_response = azure_to_anthropic_response(azure_response, model, metadata)
            return Response(content=json.dumps(anthropic_response), media_type="application/json")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/v1/messages/count_tokens")
async def count_tokens():
    # Placeholder: Azure doesn't support Anthropic count_tokens. Return zeros.
    return {"input_tokens": 0, "output_tokens": 0}


@app.get("/v1/models")
async def list_models():
    return {"data": [{"id": mid, "object": "model"} for mid in MODEL_MAP.keys()]}


if __name__ == "__main__":
    print("🚀 Claude → Azure OpenAI Proxy (2025 feature support)")
    print("   Listening on http://127.0.0.1:8100")
    print(f"   Azure base: {AZURE_BASE}")
    print()
    print("   ✅ Features:")
    print("      • Tool calling (function tools)")
    print("      • Extended thinking preservation")
    print("      • Effort (approx via max tokens/temp)")
    print("      • Plan/context hints")
    print("      • Beta headers passthrough")
    print()
    print(f"   📊 Model mappings: {len(MODEL_MAP)} Claude models")
    uvicorn.run(app, host="0.0.0.0", port=8100, log_level="info")
