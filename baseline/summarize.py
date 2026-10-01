"""Summarize tau2 runs: pass^k, the per-task success-count histogram the headroom estimate
reads, and per-episode turn, token and time averages.

Usage: python summarize.py <run_dir_or_results.json> [...]
"""

import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path


def is_successful(reward: float) -> bool:
    # Same tolerance as tau2.metrics.agent_metrics.is_successful.
    return abs(reward - 1) <= 1e-6


def pass_hat_k(n: int, c: int, k: int) -> float:
    return math.comb(c, k) / math.comb(n, k)


def usage_sum(messages, role, field):
    return sum((m.get("usage") or {}).get(field) or 0 for m in messages if m["role"] == role)


def summarize(path: Path) -> dict:
    results_file = path / "results.json" if path.is_dir() else path
    sims = json.loads(results_file.read_text())["simulations"]

    # Episodes that died on infrastructure errors carry no reward_info. tau2 drops them from
    # its own metrics; count them here so the exclusion is visible next to the score.
    infra = [s for s in sims if not s.get("reward_info")]
    sims = [s for s in sims if s.get("reward_info")]

    successes = defaultdict(int)
    trials = defaultdict(int)
    for s in sims:
        trials[s["task_id"]] += 1
        successes[s["task_id"]] += is_successful(s["reward_info"]["reward"])

    n = min(trials.values())
    name = path.name if path.is_dir() else path.stem
    row = {"run": name, "tasks": len(trials), "trials": n}
    if infra:
        # A task whose trials are unequal after exclusions has a pass^k over a smaller n:
        # flag it, because pass^k is only comparable across runs at the same n.
        row["infra_dropped"] = f"{len(infra)} episodes ({100*len(infra)/(len(sims)+len(infra)):.0f}%)"
        row["trials_range"] = f"{min(trials.values())}-{max(trials.values())}"
    for k in range(1, n + 1):
        row[f"pass^{k}"] = 100 * sum(pass_hat_k(trials[t], successes[t], k) for t in trials) / len(trials)

    # Headroom. pass^k is an average over each task's
    # success count, so the histogram of that count, not the score, says how much room RL has.
    # The band 0.6 <= p < 1 is the fuel: tasks the model usually solves but not reliably. It is
    # also where GRPO groups come out mixed, since all-success and all-failure groups both give
    # zero advantage, so the same number sizes the training set.
    rates = {t: successes[t] / trials[t] for t in trials}
    band = {t for t, p in rates.items() if 0.6 <= p < 1}
    row["never_solved"] = sum(p == 0 for p in rates.values())
    row["mixed"] = sum(0 < p < 1 for p in rates.values())
    row["band_0.6-1.0"] = len(band)
    row["always_solved"] = sum(p == 1 for p in rates.values())
    # Infra drops leave some tasks with fewer trials, and their counts aren't on the same scale,
    # so the histogram covers the full-n tasks only. The band counts above use the rate instead
    # and cover every task.
    n_full = max(trials.values())
    full = Counter(successes[t] for t in trials if trials[t] == n_full)
    row["c_hist"] = {c: full.get(c, 0) for c in range(n_full + 1)}
    # What pass^4 would be if RL took the band to p=0.95 and changed nothing else.
    # A plug-in estimate on measured rates, so it is an optimistic ceiling, not a
    # measurement, and it is optimistic wherever a task's variance is the customer's, not the
    # agent's.
    row["pass^4_if_band@.95"] = 100 * sum(
        (0.95 if t in band else rates[t]) ** 4 for t in trials) / len(trials)

    per_ep = len(sims)
    row["agent_turns"] = sum(sum(m["role"] == "assistant" for m in s["messages"]) for s in sims) / per_ep
    row["agent_prompt_tok"] = sum(usage_sum(s["messages"], "assistant", "prompt_tokens") for s in sims) / per_ep
    row["agent_gen_tok"] = sum(usage_sum(s["messages"], "assistant", "completion_tokens") for s in sims) / per_ep
    row["sec/episode"] = sum(s.get("duration") or 0 for s in sims) / per_ep
    row["endings"] = dict(Counter(s["termination_reason"] for s in sims))
    return row


def render_histogram(hist, band, tasks):
    """Draw the success-count distribution and say how much headroom it shows.

    The band 0.6 <= c/n < 1 is the headroom: taking ~30 such tasks from p~0.75 to p~0.95 is
    worth ~+13 pass^4. The thresholds (25 and 15 tasks) are for
    114 tasks, scaled when a run covers a different number of tasks.
    """
    n, total = max(hist), sum(hist.values())
    lines = [f"success histogram (c of {n} trials, {total} tasks)"]
    for c, count in hist.items():
        note = "  zero-gradient" if c in (0, n) else ""
        bar = "#" * round(40 * count / total) if total else ""
        lines.append(f"  {c}/{n}  {bar:<40} {count:>4} ({100 * count / total:5.1f}%){note}")

    # Smoke runs and other tiny splits scale the thresholds down to nothing, which would let
    # an empty band read as enough. Report the count and stop.
    if tasks < 40:
        lines.append(f"  band of {band} tasks -- split too small for a headroom verdict")
        return lines

    hi, lo = round(25 * tasks / 114), round(15 * tasks / 114)
    if band >= hi:
        verdict = "enough headroom for a pass^4 gain"
    elif band >= lo:
        verdict = "thin headroom: a pass^4 gain needs nearly the whole band"
    else:
        verdict = "too little headroom for a pass^4 gain"
    scale = "" if tasks == 114 else f" [thresholds scaled from 114 tasks: {hi}/{lo}]"
    lines.append(f"  band of {band}: {verdict}{scale}")
    return lines


def main():
    rows = [summarize(Path(p)) for p in sys.argv[1:]]
    for r in rows:
        print(f"\n{r.pop('run')}")
        hist, band, tasks = r.pop("c_hist"), r["band_0.6-1.0"], r["tasks"]
        for key, val in r.items():
            if isinstance(val, float):
                val = f"{val:.1f}"
            print(f"  {key:18} {val}")
        print()
        for line in render_histogram(hist, band, tasks):
            print(f"  {line}")


if __name__ == "__main__":
    main()
