#!/usr/bin/env bash
# Baseline eval: one Qwen3.5 size on a τ³ split, with thinking
# off and then on. Official protocol, matching Sierra's leaderboard runs: gpt-5.2 user
# simulator at low reasoning, seed 300, max_steps 200, and TRIALS trials (default 4).
#
# Needs: ./serve.sh <same size> running, OPENAI_API_KEY set (the user simulator is gpt-5.2).
# Usage: ./run_baseline.sh 4B          full run (2 modes x 114 tasks x 4 trials = 912 episodes)
#        SMOKE=1 ./run_baseline.sh 4B  2 tasks x 1 trial per mode, to check the wiring first
#        SPLIT=test ./run_baseline.sh 4B   the 40-task split only (the first baselines used this)
#        MODES=off ./run_baseline.sh 0.8B  one thinking mode only (off is settled; halves the cost)
#        DOMAIN=airline ./run_baseline.sh 4B   transfer baseline: all 50 airline tasks
#        CONC=40 TRIALS=8 MODES=off SPLIT=base ./run_baseline.sh 2B   the 114-task re-baseline;
#          8 trials both resolve the per-task success count and make this the final eval's
#          base arm, which the final eval reuses
set -euo pipefail
cd "$(dirname "$0")"
SIZE=${1:?usage: run_baseline.sh 2B|4B}
DOMAIN=${DOMAIN:-retail}  # DOMAIN=airline or telecom to override
# base = all 114 retail tasks. Since Sept 20 nothing in tau2-bench is trained on: training data
# is the converted tau-bench 500, so the whole official set is held out and evals are directly
# comparable to the leaderboard. 40 tasks was too noisy (see baseline-results.md).
SPLIT=${SPLIT:-base}
# 8 for anything that feeds a reported number: pass^k is an average over each task's success count,
# and n=8 resolves that count into 9 bins instead of 5. 4 stays the default so the first
# baseline runs reproduce.
TRIALS=${TRIALS:-4}
# The agent waits on the gpt-5.2 customer far more than it generates (first baselines: 1-2 running
# requests, 0.8% KV cache), so the GPU is not the limit — OpenAI's rate limit is. Raising this
# shortens the run, which on Spot also shrinks the window a preemption can land in. 16 stays
# the default so the first baseline runs reproduce.
CONC=${CONC:-16}
# tau2 reads tau2-bench/.env itself, but this guard (and litellm, when the key is only in
# the file) runs first, so load it here too.
[ -f tau2-bench/.env ] && { set -a; . tau2-bench/.env; set +a; }
[ -f ../.env.deepinfra ] && { set -a; . ../.env.deepinfra; set +a; }
: "${OPENAI_API_KEY:?set OPENAI_API_KEY in the environment or in tau2-bench/.env (user simulator is gpt-5.2)}"
# The customer. Official protocol is gpt-5.2 at low reasoning; USER_LLM swaps it (the
# customer-comparison arms). Any non-default customer gets its own results directory via
# USER_TAG, or --auto-resume would silently continue the gpt-5.2 run's results.json.
USER_LLM=${USER_LLM:-gpt-5.2}
USER_ARGS=${USER_ARGS:-'{"reasoning_effort": "low"}'}
USER_TAG=${USER_TAG:-}
[ "$USER_LLM" != gpt-5.2 ] && [ -z "$USER_TAG" ] && { echo "set USER_TAG for a non-default USER_LLM" >&2; exit 2; }

# Trained arm: ARM names a LoRA adapter served as qwen3.5-<SIZE>-<ARM> by serve-eval.sh;
# it goes into the agent model name and every output directory, so a trained run can never
# --auto-resume into the base arm's results. MAX_TOKENS caps one agent reply. The frozen base
# arm ran uncapped; its longest reply was 5,967 tokens (results/base_gpt52), so a cap above that leaves
# every base episode unchanged and still bounds a runaway spiral. Both unset = the base protocol.
ARM=${ARM:-}
AGENT="qwen3.5-${SIZE}${ARM:+-$ARM}"
MAX_TOKENS=${MAX_TOKENS:-}

# Sampling: Qwen3.5 model card, "general tasks" settings for each mode.
# Thinking-on matches Sierra's Qwen3.5-397B leaderboard run (temperature 1.0, top_p 0.95).
BASE='"api_base": "http://localhost:8000/v1", "presence_penalty": 1.5'
[ -n "$MAX_TOKENS" ] && BASE="$BASE, \"max_tokens\": $MAX_TOKENS"
THINK_ON="{$BASE, \"temperature\": 1.0, \"top_p\": 0.95, \"extra_body\": {\"top_k\": 20, \"chat_template_kwargs\": {\"enable_thinking\": true}}}"
THINK_OFF="{$BASE, \"temperature\": 0.7, \"top_p\": 0.8, \"extra_body\": {\"top_k\": 20, \"chat_template_kwargs\": {\"enable_thinking\": false}}}"

if [ "${SMOKE:-0}" = 1 ]; then
  SCOPE=(--num-tasks 2 --num-trials 1); TAG=smoke
else
  SCOPE=(--num-trials "$TRIALS"); TAG=baseline
  # tau2 resumes from an existing results.json, so a run at a different trial count needs its
  # own directory or it would silently reuse the old one at the wrong n. The first baselines' dirs were
  # 4-trial and predate this suffix, so 4 keeps the original name. This naming is also what
  # makes --auto-resume below safe: a resume can only ever land in a directory whose size,
  # domain, split, mode and trial count already match.
  [ "$TRIALS" = 4 ] || TAG="baseline_n${TRIALS}"
fi
# Outside the branch so smoke runs are separated by customer too (else --auto-resume reuses them).
[ -n "$USER_TAG" ] && TAG="${TAG}_user-${USER_TAG}"

# The first baselines settled thinking mode (off: same score, 1.6x faster), so later runs need only one mode.
for MODE in ${MODES:-off on}; do
  if [ "$MODE" = on ]; then ARGS=$THINK_ON; else ARGS=$THINK_OFF; fi
  (cd tau2-bench && ../.venv-tau2/bin/tau2 run \
    --domain "$DOMAIN" --task-split-name "$SPLIT" \
    --agent-llm "hosted_vllm/${AGENT}" --agent-llm-args "$ARGS" \
    --user-llm "$USER_LLM" --user-llm-args "$USER_ARGS" \
    --seed 300 --max-steps 200 --max-concurrency "$CONC" \
    --auto-resume \
    "${SCOPE[@]}" \
    --save-to "${TAG}_${AGENT}_${DOMAIN}-${SPLIT}_think-${MODE}")
done

.venv-tau2/bin/python summarize.py tau2-bench/data/simulations/${TAG}_${AGENT}_${DOMAIN}-${SPLIT}_think-*
