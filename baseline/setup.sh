#!/usr/bin/env bash
# One-time setup on the GPU box (Linux + CUDA, 1x A100 80GB Spot on GCE).
# Three venvs: vLLM and tau2 pin different litellm/openai/pydantic versions, and the trainer
# needs its own torch/transformers that must not disturb the vLLM server's.
set -euo pipefail
cd "$(dirname "$0")"
# TRAIN_ONLY=1 skips straight to the trainer venv (session.sh uses this after a preemption).
TRAIN_ONLY=${TRAIN_ONLY:-0}
export PATH="$HOME/.local/bin:/usr/local/cuda-12.9/bin:$PATH"

TAU2_COMMIT=b7ea907  # sierra-research/tau2-bench HEAD on 2026-09-17 = τ³ v1.0.1

if [ "$TRAIN_ONLY" != 1 ]; then
# System build tools the GCE Deep Learning VM image lacks:
#   python3.12-dev  Triton compiles cuda_utils.c against Python headers at first run
#   ninja-build     FlashInfer JIT-builds its sampling kernel and shells out to ninja
sudo apt-get update -qq
sudo apt-get install -y -qq python3.12-dev build-essential ninja-build

command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# vLLM server: >=0.29.0 is required for Qwen3.5 (partial-fused-LoRA fix; text-only loading fix in 0.27).
uv venv --python 3.12 .venv-vllm
uv pip install --python .venv-vllm/bin/python "vllm>=0.29.0"

# tau2 client, pinned to the commit every run uses.
[ -d tau2-bench ] || git clone https://github.com/sierra-research/tau2-bench.git
git -C tau2-bench checkout "$TAU2_COMMIT"
uv venv --python 3.12 .venv-tau2
uv pip install --python .venv-tau2/bin/python -e tau2-bench
# tau2 imports its voice module at startup, but websockets only ships in the optional
# [voice] extra (b7ea907). Install it alone: the full extra needs pyaudio, which wants
# system audio libraries this box doesn't have.
uv pip install --python .venv-tau2/bin/python "websockets>=13.0"
fi

# Trainer.
#   transformers>=5.2           first release with Qwen3.5
#   flash-linear-attention      DeltaNet chunked kernels (imports as `fla`)
#   causal-conv1d               DeltaNet short conv
# Without the last two Transformers SILENTLY falls back to a pure-torch path (~50 s/step), so
# check_train_env.py below fails the setup instead of letting that surface as a slow run.
# causal-conv1d compiles against the installed torch, hence torch first and no build isolation.
# torch comes from the cu129 index: the default wheel is built for CUDA 13.0, the image's
# toolkit is 12.9, and torch refuses to compile an extension across that mismatch
# (found Sept 23). The vLLM venv keeps its own cu130 torch; the 580 driver runs both.
export CUDA_HOME=/usr/local/cuda-12.9 MAX_JOBS=12 TORCH_CUDA_ARCH_LIST=8.0
[ -d .venv-train ] || uv venv --python 3.12 .venv-train
uv pip install --python .venv-train/bin/python "transformers>=5.2" peft accelerate
uv pip install --python .venv-train/bin/python --reinstall torch \
  --index-url https://download.pytorch.org/whl/cu129
uv pip install --python .venv-train/bin/python flash-linear-attention
uv pip install --python .venv-train/bin/python --no-build-isolation causal-conv1d

if [ "$TRAIN_ONLY" != 1 ]; then
.venv-vllm/bin/python -c "import vllm; print('vllm', vllm.__version__)"
.venv-tau2/bin/tau2 --help >/dev/null && echo "tau2 ok ($(git -C tau2-bench rev-parse --short HEAD))"
fi
.venv-train/bin/python check_train_env.py
