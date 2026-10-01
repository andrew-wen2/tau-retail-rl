#!/usr/bin/env bash
# Serve one Qwen3.5 size on :8000 with vLLM's OpenAI-compatible API.
# Flags follow the Qwen3.5 model card: qwen3 reasoning parser (splits <think> out of the
# content), qwen3_coder tool parser, text-only (skips the vision tower).
# Usage: ./serve.sh 4B   (or 2B)
set -euo pipefail
cd "$(dirname "$0")"
SIZE=${1:?usage: serve.sh 2B|4B}

exec .venv-vllm/bin/vllm serve "Qwen/Qwen3.5-${SIZE}" \
  --served-model-name "qwen3.5-${SIZE}" \
  --port 8000 \
  --max-model-len 131072 \
  --language-model-only \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder
