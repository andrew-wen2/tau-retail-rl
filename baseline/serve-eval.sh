#!/usr/bin/env bash
# Serve a trained adapter for the final eval: serve.sh's flags unchanged (same max model
# length, parsers, stock chat template and default GPU memory), plus LoRA. serve.sh stays
# byte-identical because the frozen base arm ran against it (see serve-train.sh's header).
#
# The adapter is served as qwen3.5-<SIZE>-<ARM>, the name run_baseline.sh builds from ARM.
# Base requests (qwen3.5-<SIZE>) still reach the plain model, which final_eval.sh uses to check
# the adapter actually changes outputs.
#
# Usage: ./serve-eval.sh 2B soupm ../runs/soup/adapters/003-soupm/serve
set -euo pipefail
cd "$(dirname "$0")"
SIZE=${1:?usage: serve-eval.sh SIZE ARM ADAPTER_DIR}
ARM=${2:?usage: serve-eval.sh SIZE ARM ADAPTER_DIR}
ADAPTER=${3:?usage: serve-eval.sh SIZE ARM ADAPTER_DIR}
[ -f "$ADAPTER/adapter_config.json" ] || { echo "not an adapter dir: $ADAPTER" >&2; exit 1; }

exec .venv-vllm/bin/vllm serve "Qwen/Qwen3.5-${SIZE}" \
  --served-model-name "qwen3.5-${SIZE}" \
  --port 8000 \
  --max-model-len 131072 \
  --language-model-only \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --enable-lora --max-lora-rank "${LORA_RANK:-32}" \
  --lora-modules "qwen3.5-${SIZE}-${ARM}=$(cd "$ADAPTER" && pwd)"
