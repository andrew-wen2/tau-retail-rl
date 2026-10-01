#!/usr/bin/env bash
# Final eval, VM side: the trained arm on all 114 held-out retail tasks x 8 trials under
# the base arm's protocol (gpt-5.2 customer at low reasoning, seed 300, max_steps 200, stock
# template), paired later against the frozen base arm. Idempotent: after a preemption, re-run it
# and tau2 --auto-resume continues the same results.json.
#
# Runaway guards: MAX_TOKENS caps each agent reply (base arm max was 5,967, so 8192 changes no base
# episode); eval_watchdog.py kills the run on spend, cost per episode, wall clock or stale progress.
# Spend lands in the driver ledger (api_spend.jsonl) as customer cost x 1.45 for the judge.
#
# Usage (on the VM, from ~): API_BUDGET_USD=<guard> baseline/final_eval.sh
#   Another customer: add USER_LLM, USER_ARGS, USER_TAG (as run_baseline.sh), TRIALS,
#   and the cost model JUDGE_FACTOR / PER_EP_FIXED_USD / MAX_PER_EP_USD for the watchdog and ledger.
set -uo pipefail
cd ~/baseline
ARM=${ARM:-soupm}
ADAPTER=${ADAPTER:-$HOME/runs/soup/adapters/003-soupm/serve}
MAX_TOKENS=${MAX_TOKENS:-8192}
GUARD=${API_BUDGET_USD:?set API_BUDGET_USD (the ledger guard)}
TRIALS=${TRIALS:-8}
USER_TAG=${USER_TAG:-}
export USER_LLM USER_ARGS USER_TAG
export JUDGE_FACTOR=${JUDGE_FACTOR:-1.45} PER_EP_FIXED_USD=${PER_EP_FIXED_USD:-0}
# One state dir per (customer, adapter): it holds the smoke marker and the spend already recorded.
STATE=$HOME/runs/final_eval${USER_TAG:+_$USER_TAG}; [ "$ARM" = soupm ] || STATE=${STATE}_$ARM
mkdir -p "$STATE"
MODEL="qwen3.5-2B-${ARM}"
T=.venv-tau2/bin/python
[ -f tau2-bench/.env ] && { set -a; . tau2-bench/.env; set +a; }
log() { echo "$(date +%H:%M) $*"; }

# Spend already recorded for this eval, so a relaunch appends only the difference.
ledger_add() {  # $1 = total estimated spend so far for the stage, $2 = stage name
  local f="$STATE/recorded_$2" prev; prev=$(cat "$f" 2>/dev/null || echo 0)
  $T - "$1" "$prev" "$2" "$ARM" <<'EOF'
import sys; sys.path.insert(0, "..")
from driver import record_spend
total, prev, stage, arm = float(sys.argv[1]), float(sys.argv[2]), sys.argv[3], sys.argv[4]
if total > prev: record_spend(total - prev, f"final eval {arm} {stage} (customer + estimated judge)")
EOF
  echo "$1" > "$f"
}
est_usd() {  # estimated spend of one tau2 run dir: customer x JUDGE_FACTOR + episodes x PER_EP_FIXED_USD
  $T - "$1" <<'EOF'
import json, sys
from pathlib import Path
d = Path(sys.argv[1]); c = {}
m = d / "results.json"
if m.exists():
    for s in json.loads(m.read_text()).get("simulations") or []: c[s["id"]] = s.get("user_cost") or 0
for f in (d / "simulations").glob("*.json") if (d / "simulations").is_dir() else []:
    c.setdefault(f.stem, json.loads(f.read_text()).get("user_cost") or 0)
import os
print(round(sum(c.values()) * float(os.environ["JUDGE_FACTOR"]) + len(c) * float(os.environ["PER_EP_FIXED_USD"]), 4))
EOF
}
spent() { $T -c 'import sys; sys.path.insert(0, ".."); from driver import spent_usd; print(round(spent_usd(), 4))'; }

# 1. Server: serve.sh's config plus the adapter. Anything else on :8000 (the training server) goes.
if ! curl -sf localhost:8000/v1/models | grep -q "\"$MODEL\""; then
  pkill -f "[v]llm serve" && sleep 15
  log "starting serve-eval.sh ($ADAPTER)"
  setsid nohup ./serve-eval.sh 2B "$ARM" "$ADAPTER" > serve-eval.log 2>&1 < /dev/null &
  for _ in $(seq 120); do curl -sf localhost:8000/health >/dev/null && break; sleep 5; done
  curl -sf localhost:8000/v1/models | grep -q "\"$MODEL\"" || { log "server did not come up"; tail -30 serve-eval.log; exit 1; }
fi

# 2. The adapter must change the model: same prompt, greedy, logprobs from base vs adapter.
$T - "$MODEL" <<'EOF' || { log "ADAPTER CHECK FAILED"; exit 1; }
import json, sys, urllib.request
def lp(model):
    body = {"model": model, "prompt": "Customer: I want to return the headphones from order #W2378156.\nAgent:",
            "max_tokens": 16, "temperature": 0, "logprobs": 1}
    r = urllib.request.Request("http://localhost:8000/v1/completions", json.dumps(body).encode(),
                               {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(r, timeout=120).read())["choices"][0]["logprobs"]["token_logprobs"]
a, b = lp("qwen3.5-2B"), lp(sys.argv[1])
diff = max(abs(x - y) for x, y in zip(a, b))
print(f"adapter check: max |logprob diff| over {len(a)} tokens = {diff:.4f}")
sys.exit(0 if diff > 1e-3 else 1)
EOF

# 3. Refuse to resume onto a NUL-damaged checkpoint (Spot preemption, see spot-preemption notes).
TAG=baseline; [ "$TRIALS" = 4 ] || TAG=baseline_n$TRIALS; UT=${USER_TAG:+_user-$USER_TAG}
RUN_DIR=tau2-bench/data/simulations/${TAG}${UT}_${MODEL}_retail-base_think-off
if [ -d "$RUN_DIR" ] && grep -qaP '\x00' "$RUN_DIR"/results.json 2>/dev/null; then
  log "results.json has NUL bytes; repair before resuming"; exit 1
fi

# 4. Smoke: 2 tasks x 1 trial through the adapter, once.
SMOKE_DIR=tau2-bench/data/simulations/smoke${UT}_${MODEL}_retail-base_think-off
if [ ! -f "$STATE/smoke_ok" ]; then
  log "smoke (2 tasks x 1 trial)"
  SMOKE=1 ARM=$ARM MAX_TOKENS=$MAX_TOKENS MODES=off ./run_baseline.sh 2B < /dev/null > "$STATE/smoke.log" 2>&1
  ledger_add "$(est_usd "$SMOKE_DIR")" smoke
  $T - "$SMOKE_DIR" <<'EOF' || { log "SMOKE FAILED (see $STATE/smoke.log)"; exit 1; }
import json, sys
sims = json.load(open(sys.argv[1] + "/results.json"))["simulations"]
ok = [s for s in sims if s.get("reward_info")]
print(f"smoke: {len(ok)}/{len(sims)} scored, rewards {[s['reward_info']['reward'] for s in ok]}")
sys.exit(0 if len(sims) == 2 and len(ok) == 2 else 1)
EOF
  touch "$STATE/smoke_ok"
fi

# 5. The run, under the watchdog. Ceiling = guard minus everything already in the ledger, plus
#    this run's own spend that an earlier launch already recorded (the watchdog counts it again).
prev=$(cat "$STATE/recorded_main" 2>/dev/null || echo 0)
CEIL=$($T -c "print(round($GUARD - $(spent) + $prev, 2))")
log "main run: ceiling \$$CEIL (guard \$$GUARD, ledger \$$(spent))"
CONC=40 TRIALS=$TRIALS MODES=off SPLIT=base ARM=$ARM MAX_TOKENS=$MAX_TOKENS \
  setsid ./run_baseline.sh 2B < /dev/null >> "$STATE/main.log" 2>&1 &
PG=$!
sleep 30
CEILING_USD=$CEIL $T eval_watchdog.py "$RUN_DIR" "$PG" >> "$STATE/watchdog.log" 2>&1
WD=$?
wait "$PG"; RC=$?
ledger_add "$(est_usd "$RUN_DIR")" main
log "ledger now \$$(spent)"
[ "$WD" = 3 ] && { log "KILLED: $(tail -1 "$STATE/watchdog.log")"; exit 3; }
[ "$RC" = 0 ] || { log "run exited $RC"; tail -20 "$STATE/main.log"; exit 1; }
tail -40 "$STATE/main.log"
log "FINAL-DONE"
