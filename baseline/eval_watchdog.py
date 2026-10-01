"""Spend and runaway watchdog for a tau2 eval run. Kills the run's process group when:

  1. estimated spend passes CEILING_USD (the ledger guard minus what the ledger already holds),
  2. after MIN_EPISODES, the mean customer cost per episode passes MAX_PER_EP_USD,
  3. the run has been going longer than MAX_HOURS, or
  4. no progress can be read for STALE_MIN minutes while the run is alive (fail closed).

tau2 records only the customer's cost (`user_cost`); the gpt-4.1 NL-assertion judge has no cost field
anywhere. With the gpt-5.2 customer it measured ~45% on top of customer cost (JUDGE_FACTOR 1.45);
with a near-free customer it is better modelled per episode (PER_EP_FIXED_USD). Reads both of
tau2's checkpoint formats: monolithic results.json and results.json + simulations/*.json.

Usage: python eval_watchdog.py RUN_DIR PGID  (settings from env; exits when the group is gone)
Exit: 0 run ended on its own, 3 killed by a rule (reason on the last stdout line)."""
import json, os, signal, sys, time
from pathlib import Path

RUN_DIR, PGID = Path(sys.argv[1]), int(sys.argv[2])
CEILING_USD = float(os.environ["CEILING_USD"])
JUDGE_FACTOR = float(os.environ.get("JUDGE_FACTOR", "1.45"))
PER_EP_FIXED_USD = float(os.environ.get("PER_EP_FIXED_USD", "0"))
MAX_PER_EP_USD = float(os.environ.get("MAX_PER_EP_USD", "0.0376"))  # 2x base arm's customer cost
MIN_EPISODES = int(os.environ.get("MIN_EPISODES", "100"))
MAX_HOURS = float(os.environ.get("MAX_HOURS", "3"))
STALE_MIN = float(os.environ.get("STALE_MIN", "20"))
POLL_S = 60

_costs: dict[str, float] = {}  # simulation id -> user_cost, cached across polls


def read_costs() -> dict[str, float]:
    meta = RUN_DIR / "results.json"
    if meta.exists():
        for s in json.loads(meta.read_text()).get("simulations") or []:
            _costs[s["id"]] = s.get("user_cost") or 0.0
    sims = RUN_DIR / "simulations"
    if sims.is_dir():
        for f in sims.glob("*.json"):
            if f.stem not in _costs:
                _costs[f.stem] = json.loads(f.read_text()).get("user_cost") or 0.0
    return _costs


def alive() -> bool:
    """Any non-zombie process left in the group. killpg(0) alone is not enough: final_eval.sh
    reaps the run only after this script exits, so a finished run lingers as a zombie leader."""
    if not Path("/proc").is_dir():  # not Linux: best effort
        try:
            os.killpg(PGID, 0)
            return True
        except (ProcessLookupError, PermissionError):
            return False
    for stat in Path("/proc").glob("[0-9]*/stat"):
        try:
            f = stat.read_text().rsplit(")", 1)[1].split()
        except OSError:
            continue
        if int(f[2]) == PGID and f[0] != "Z":  # fields after "(comm)": state ppid pgrp
            return True
    return False


def kill(reason: str) -> None:
    print(f"WATCHDOG KILL: {reason}", flush=True)
    for sig in (signal.SIGTERM, signal.SIGKILL):
        if not alive():
            break
        try:
            os.killpg(PGID, sig)
        except (ProcessLookupError, PermissionError):
            break
        time.sleep(20)
    sys.exit(3)


start = last_ok = time.time()
while alive():
    time.sleep(POLL_S)
    try:
        costs = read_costs()
        last_ok = time.time()
    except (OSError, ValueError, KeyError) as e:  # mid-rewrite or NUL tail: retry next poll
        print(f"read failed: {e!r}", flush=True)
        if (time.time() - last_ok) / 60 > STALE_MIN:
            kill(f"no readable progress for {STALE_MIN:.0f} min")
        continue
    n, cust = len(costs), sum(costs.values())
    est = cust * JUDGE_FACTOR + n * PER_EP_FIXED_USD
    per_ep = cust / n if n else 0.0
    print(f"{time.strftime('%H:%M')} episodes {n} customer ${cust:.2f} est ${est:.2f} "
          f"/ ceiling ${CEILING_USD:.2f} per-ep ${per_ep:.4f}", flush=True)
    if est > CEILING_USD:
        kill(f"estimated spend ${est:.2f} > ceiling ${CEILING_USD:.2f}")
    if n >= MIN_EPISODES and per_ep > MAX_PER_EP_USD:
        kill(f"customer cost ${per_ep:.4f}/episode > ${MAX_PER_EP_USD} after {n} episodes")
    if (time.time() - start) / 3600 > MAX_HOURS:
        kill(f"wall clock over {MAX_HOURS} h")
print("run ended", flush=True)
