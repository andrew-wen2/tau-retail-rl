#!/usr/bin/env bash
# Profiling pass over all 418 converted tasks at n=8 with the base model, split into chunks
# at rising concurrency so the same episodes also give the throughput sweep (design Open
# Question 4). Resumable at task granularity after a preemption. Run from the repo root
# (the VM home), detached:  setsid nohup baseline/profile.sh > profile.log 2>&1 < /dev/null &
set -euo pipefail
cd "$(dirname "$0")/.."
PY=baseline/.venv-tau2/bin/python
# OUT/MAN/PROFILE are overridable so a re-profile (e.g. after a customer change) does not read
# the old records as already done: OUT=records/profile-v2 MAN=runs/profile-v2 PROFILE=profile-v2.json
OUT=${OUT:-records/profile}
MAN=${MAN:-runs/profile}
PROFILE=${PROFILE:-profile.json}
TASKS_FILE=${TASKS_FILE:-tb500_retail.json}   # tb500_hard.json + SPLIT_FILE=split_hard.json
SPLIT_FILE=${SPLIT_FILE:-split_tb500.json}
export OUT MAN TASKS_FILE SPLIT_FILE
mkdir -p "$MAN" "$OUT"
# Manifests are rebuilt on every launch from what is already on disk: a task with a full
# group of 8 in records/profile is done and is left out, so a relaunch after a preemption
# re-runs only the unfinished tasks. Chunk b10x.json is removed once it has nothing left.
$PY - <<'PYEOF'
import glob, json, os, collections
OUT, MAN = os.environ["OUT"], os.environ["MAN"]
TASKS_FILE, SPLIT_FILE = os.environ["TASKS_FILE"], os.environ["SPLIT_FILE"]
ids = json.load(open("tasks/" + SPLIT_FILE))["all"]
have = collections.Counter(os.path.basename(p).split("-", 2)[2].rsplit("-k", 1)[0]
                           for p in glob.glob(f"{OUT}/*.json.gz"))
done = {t for t, n in have.items() if n >= 8}
plan = [(100, 0, 48, 32), (101, 48, 112, 64), (102, 112, 192, 128), (103, 192, len(ids), 96)]
for b, lo, hi, conc in plan:
    todo = [t for t in ids[lo:hi] if t not in done]
    path = f"{MAN}/b{b}.json"
    if not todo:
        if os.path.exists(path): os.remove(path)
        continue
    m = {"batch_id": b, "policy_version": "base-b7ea907", "model": "qwen3.5-2B",
         "tasks": todo, "G": 8, "seed": 300 + b + 1000 * len(done), "out_dir": OUT,
         "concurrency": conc, "guards": {"max_steps": 80}, "tasks_file": TASKS_FILE}
    json.dump(m, open(path, "w"), indent=1)
    print(f"chunk {b}: {len(todo)} tasks to run")
PYEOF
for b in 100 101 102 103; do
  [ -f "$MAN/b$b.json" ] || continue
  echo "$(date +%T) batch $b"
  $PY driver.py batch "$MAN/b$b.json" 2>/dev/null | tail -1
done
$PY records.py profile "$OUT" > "$PROFILE"
$PY records.py gate "$OUT" | head -11
echo "done $(date +%T)"
