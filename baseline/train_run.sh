#!/usr/bin/env bash
# The first full training run. Value-weighted sampling (P(mixed) weighting, Sept 26) over all
# 402 hardened train tasks (prior from profile-hard.json), DeepSeek customer, 32 tasks x G=8 =
# 256 episodes/step, the 60 hardened val tasks x 8 every VAL_EVERY steps (twice at step 0: the
# noise floor). p4-b ran its first 15 steps on the 220 profile-mixed tasks with val x 4.
# Run from the repo root (the VM home), detached:
#   RUN=p4-b setsid nohup baseline/train_run.sh > train_run.log 2>&1 < /dev/null &
# Re-running after a preemption or the Flex 6h stop resumes from the last checkpoint; run
# baseline/verify_resume.sh first.
set -euo pipefail
cd "$(dirname "$0")/.."
RUN=${RUN:-p4-b}
STEPS=${STEPS:-40}
VAL_EVERY=${VAL_EVERY:-5}
TR=baseline/.venv-train/bin/python
# The driver refuses a batch once the spend ledger reaches this.
export API_BUDGET_USD=${API_BUDGET_USD:?set API_BUDGET_USD (the spend guard)}

if [ ! -f runs/$RUN.config.json ]; then
  $TR - <<PYEOF
import json
json.dump({"sampler": "value", "steps": $STEPS, "G": 8, "episodes_per_step": 256,
           "concurrency": 128, "sampler_weight": "mixed", "sampler_pool": "all",
           "tasks_file": "tb500_hard.json", "split_file": "split_hard.json",
           "profile_file": "profile-hard.json", "val_every": $VAL_EVERY, "val_G": 8,
           "val_repeat0": True}, open("runs/$RUN.config.json", "w"), indent=1)
PYEOF
fi

RESUME=""
[ -f runs/$RUN/ckpt/state.pt ] && RESUME=--resume
$TR trainer.py run --run $RUN --config runs/$RUN.config.json $RESUME
echo "done $(date +%T)"
