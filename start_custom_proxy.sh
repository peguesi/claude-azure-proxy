#!/usr/bin/env bash
set -euo pipefail

# Activate venv if present
test -f "$HOME/.venv/bin/activate" && source "$HOME/.venv/bin/activate"

if [ -z "${AZURE_OPENAI_KEY:-}" ] || [ -z "${AZURE_OPENAI_BASE:-}" ]; then
  echo "❌ AZURE_OPENAI_KEY and AZURE_OPENAI_BASE must be set"
  exit 1
fi

# Free port 8100 if occupied
if lsof -i tcp:8100 >/dev/null 2>&1; then
  lsof -i tcp:8100 | awk 'NR>1{print $2}' | xargs -r kill
fi

echo "🚀 Starting custom Claude→Azure proxy on http://127.0.0.1:8100"
exec python3 claude_azure_proxy.py
