#!/usr/bin/env bash
# Integration run: the 16-task dev slice at fixed G=8 (128 episodes/step), with one
# deliberate kill-and-resume, then the reward-goes-up check. Run from the repo root (the VM home), detached:
#   setsid nohup baseline/dev_run.sh > dev_run.log 2>&1 < /dev/null &
# Re-running after a preemption resumes the run from its last checkpoint.
set -euo pipefail
cd "$(dirname "$0")/.."
RUN=${RUN:-p3-dev16}
STEPS=${STEPS:-20}
KILL_AT=${KILL_AT:-3}         # SIGKILL the trainer mid-rollout of this step, once
TR=baseline/.venv-train/bin/python
PROFILE=${PROFILE:-profile.json}  # which profiling pass picks the dev slice

# Dev slice: 16 train-split tasks with mixed base outcomes (2..6 of 8), closest to p=0.5
# first, so the fixed-slice reward has room to move in both directions.
if [ ! -f runs/$RUN.config.json ]; then
  $TR - <<PYEOF
import json
prof = json.load(open("$PROFILE"))
train = set(json.load(open("tasks/split_tb500.json"))["train"])
cand = [(abs(p["successes"] / p["trials"] - 0.5), t) for t, p in prof.items()
        if t in train and p["trials"] >= 6 and 2 <= round(8 * p["successes"] / p["trials"]) <= 6]
cand.sort()
tasks = sorted(t for _, t in cand[:16])
assert len(tasks) == 16, f"only {len(tasks)} mixed tasks"
json.dump({"tasks": tasks, "steps": $STEPS, "G": 8, "episodes_per_step": 128,
           "concurrency": 128, "sampler": "fixed"}, open("runs/$RUN.config.json", "w"), indent=1)
print("dev slice", tasks, "base p", [round(prof[t]["successes"] / prof[t]["trials"], 2) for t in tasks])
PYEOF
fi

RESUME=""
[ -f runs/$RUN/ckpt/state.pt ] && RESUME=--resume

if [ ! -f runs/$RUN.killed ]; then
  # Once per run: go until step KILL_AT's rollout is in flight, then kill -9 everything.
  # Also runs when a preemption already forced a resume before the kill point.
  $TR trainer.py run --run $RUN --config runs/$RUN.config.json $RESUME &
  PID=$!
  # The manifest is written the moment the step's rollout starts; records only at its end.
  until [ -f runs/$RUN/manifests/b$(printf %04d $KILL_AT).json ]; do
    kill -0 $PID 2>/dev/null || { echo "trainer exited before the kill point"; exit 1; }
    sleep 10
  done
  sleep 60
  echo "$(date +%T) KILL -9 trainer and driver mid-rollout of step $KILL_AT"
  pkill -9 -f "trainer.py run --run $RUN" || true
  pkill -9 -f "runs/$RUN/manifests" || true
  touch runs/$RUN.killed
  sleep 5
  RESUME=--resume
fi

$TR trainer.py run --run $RUN --config runs/$RUN.config.json $RESUME
$TR trainer.py gate $RUN || true
echo "done $(date +%T)"
