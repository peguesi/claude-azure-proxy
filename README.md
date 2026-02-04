# Claude → Azure OpenAI Proxy

Minimal Anthropic `/v1/messages` shim that forwards to Azure chat completions. No LiteLLM required.

## Prereqs
- Python 3.10+
- `pip install fastapi uvicorn httpx`
- Env vars:
  - `AZURE_OPENAI_KEY` (required)
  - `AZURE_OPENAI_BASE` (e.g. `https://your-resource.openai.azure.com`)
  - `AZURE_API_VERSION` (optional, default `2024-10-01-preview`)

## Start the proxy
```bash
cd /Users/isaiahpegues/claude-azure-proxy
export AZURE_OPENAI_KEY="<your-key>"
export AZURE_OPENAI_BASE="https://<your-resource>.openai.azure.com"
./start_custom_proxy.sh   # listens on http://127.0.0.1:8100
```

## Use with Claude CLI
```bash
export ANTHROPIC_API_KEY=sk-dummy
export ANTHROPIC_BASE_URL=http://127.0.0.1:8100
claude --model claude-3-5-sonnet-20241022
```
If you see an auth conflict, run `claude /logout` once, then rerun.

## Quick curl test
```bash
echo '{"model":"claude-3-5-sonnet-20241022","messages":[{"role":"user","content":"ping"}]}' \
  | curl -s -X POST -H 'Content-Type: application/json' -d @- http://127.0.0.1:8100/v1/messages
```

## Model mappings (Claude alias → Azure deployment)
- claude-3-5-sonnet-20241022 → gpt-5.1
- claude-sonnet-4-5-20250929 → gpt-5.1
- claude-3-opus-20240229 → gpt-5.1
- claude-opus-4-5-20251101 → gpt-5.1
- claude-3-5-haiku-20241022 → gpt-5-nano
- claude-haiku-4-5-20251001 → gpt-5-nano

## Notes
- Converts Anthropic `max_tokens` → Azure `max_completion_tokens`.
- Uses Azure chat completions: `/openai/deployments/{deployment}/chat/completions` with `api-version` (default `2024-10-01-preview`).
- Keep your Azure key/base in env vars; nothing is hardcoded.
