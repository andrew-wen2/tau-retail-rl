#!/usr/bin/env bash
# Serve Qwen3.5 for TRAINING rollouts: serve.sh plus LoRA, a memory cap, and a pinned
# logprobs convention. Everything serve.sh passes is passed here identically.
#
# Why this is a separate file and not a flag on serve.sh:
#   serve.sh is the frozen eval path. run_baseline.sh:6 reads "Needs: ./serve.sh <same size>
#   running", the base arm is frozen against it at b7ea907/seed 300, and the final eval reuses that
#   arm rather than re-running it. Changing --gpu-memory-utilization changes KV cache size and
#   batching, so editing serve.sh would evaluate the trained arm under a different serving
#   configuration from the base arm it is paired against. serve.sh stays byte-identical.
#
# Usage: ./serve-train.sh 2B
#        GPU_MEM_UTIL=0.55 ./serve-train.sh 2B
#        ADAPTER=/path/to/adapter-dir ./serve-train.sh 2B    # preload instead of POSTing
set -euo pipefail
cd "$(dirname "$0")"
SIZE=${1:?usage: serve-train.sh 2B|4B}

# Runtime LoRA swap. The training loop unloads and reloads one adapter name per step via
# POST /v1/unload_lora_adapter and /v1/load_lora_adapter; without this the endpoints 404.
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True

# NOT MEASURED YET. The point of the memory-split measurement is to find the largest value at
# which this server and a trainer backward pass coexist on the one A100. 0.5 is a starting
# point, not a result -- overwrite this default once that measurement returns a number.
# The first baselines measured GPU KV cache usage at 0.8% with 87.2% prefix-cache hits (baseline-results.md:78),
# so the server needs far less KV than vLLM's 0.9 default reserves; the trainer needs the rest.
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.5}

# 32768, not serve.sh's 131072. vLLM refuses to start unless the KV pool can hold one
# max-length sequence, which would put a hard floor under GPU_MEM_UTIL for a context the
# episodes never approach: the 2B averages ~99k prompt tokens SUMMED over 13.5 turns
# (baseline-results.md:50), so the longest single prompt is ~10k and the full episode ~12k.
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}

# Must be >= the r of the adapter being served. Companion design pins r=32, and vLLM's default
# is 16, which fails with "ValueError: LoRA rank 32 is greater than max_lora_rank 16". This
# couples the server config to r: changing r means restarting the server.
LORA_RANK=${LORA_RANK:-32}

# One adapter live at a time. Each version is served under its OWN name (policy-<version>):
# reusing one name let the prefix cache keep serving KV/DeltaNet state computed under an older
# adapter, so rollouts lagged training by ~10 steps (Sept 25). The new
# version is loaded before the old is unloaded; max-cpu-loras 2 is room for that transition.
MAX_LORAS=${MAX_LORAS:-1}
MAX_CPU_LORAS=${MAX_CPU_LORAS:-2}

# Decided by the parity test, not by this default.
# vLLM V1 returns raw_logprobs: values before temperature, penalties and top-k/top-p. Rollouts
# run at temperature 0.7 / top_p 0.8 / top_k 20 / presence_penalty 1.5, and presence_penalty
# 1.5 is a large processor, so raw and processed diverge substantially here. What matters is
# that the rollout side and the training side describe the SAME distribution: mixing the two
# conventions is what produced large spurious mismatches in TRL #4159 and PipelineRL, and it
# would make the logged vLLM-vs-trainer gap measure sampling parameters rather than DeltaNet
# drift. Flip this to processed_logprobs if the parity test says the trainer sees the sampler's
# distribution.
LOGPROBS_MODE=${LOGPROBS_MODE:-raw_logprobs}

# Training-only chat template: stock Qwen3.5 plus one condition that keeps the empty
# <think></think> block on history turns the policy generated (driver.py marks them with
# reasoning_content=""). Without it, every prompt stops being a token prefix of the next at
# each customer turn and the episode can no longer train as one sequence. See the header of
# qwen35_train.jinja. serve.sh keeps the stock template, so the frozen eval path is untouched.
CHAT_TEMPLATE=${CHAT_TEMPLATE:-qwen35_train.jinja}

# Optional: preload an adapter at startup under the name the loop uses. Leave unset for the
# base-model recording pass (which serves no adapter) and for normal training (which POSTs the
# adapter after the first optimizer step).
LORA_ARGS=()
if [ -n "${ADAPTER:-}" ]; then
  [ -d "$ADAPTER" ] || { echo "ADAPTER is not a directory: $ADAPTER" >&2; exit 1; }
  LORA_ARGS=(--lora-modules "policy=${ADAPTER}")
fi

# Flag names change between vLLM releases. If any flag below is rejected, check
# `.venv-vllm/bin/vllm serve --help` on this box rather than guessing a replacement.
echo "serve-train: size=${SIZE} gpu_mem_util=${GPU_MEM_UTIL} max_model_len=${MAX_MODEL_LEN}" \
     "lora_rank=${LORA_RANK} logprobs_mode=${LOGPROBS_MODE} template=${CHAT_TEMPLATE}" \
     "adapter=${ADAPTER:-none}" >&2

exec .venv-vllm/bin/vllm serve "Qwen/Qwen3.5-${SIZE}" \
  --served-model-name "qwen3.5-${SIZE}" \
  --port 8000 \
  --max-model-len "$MAX_MODEL_LEN" \
  --language-model-only \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --enable-lora \
  --max-lora-rank "$LORA_RANK" \
  --max-loras "$MAX_LORAS" \
  --max-cpu-loras "$MAX_CPU_LORAS" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --logprobs-mode "$LOGPROBS_MODE" \
  --chat-template "$CHAT_TEMPLATE" \
  "${LORA_ARGS[@]}"
