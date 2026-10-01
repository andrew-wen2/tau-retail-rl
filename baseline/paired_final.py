"""Final eval numbers: trained arm vs the frozen base arm on the same 114 tasks x 8 trials.

Paired task-level bootstrap (resample tasks, keep each task's two success counts together) for
pass^1, pass^2, pass^4, plus the same on the DB check alone (no LLM judge), the per-task flip
counts, and endings. Dead episodes (no reward_info) are excluded and counted.

Usage: python paired_final.py BASE_RESULTS.json TRAINED_RESULTS.json [B=10000]"""
import json, random, sys
from collections import Counter, defaultdict
from math import comb


def load(path):
    out, dead, ends = defaultdict(list), 0, Counter()
    for s in json.load(open(path))["simulations"]:
        ends[str(s.get("termination_reason")).split(".")[-1]] += 1
        ri = s.get("reward_info")
        if not ri:
            dead += 1
            continue
        db = (ri.get("db_check") or {}).get("db_reward")
        out[s["task_id"]].append((ri["reward"] >= 1.0, db is not None and db >= 1.0))
    return out, dead, ends


def pass_k(trials, k):
    n, c = len(trials), sum(trials)
    return comb(c, k) / comb(n, k) if n >= k else float("nan")


def main():
    base, bdead, bends = load(sys.argv[1])
    tr, tdead, tends = load(sys.argv[2])
    B = int(sys.argv[3]) if len(sys.argv) > 3 else 10000
    tasks = sorted(set(base) & set(tr))
    print(f"tasks {len(tasks)} | trials base {Counter(map(len, base.values()))} trained {Counter(map(len, tr.values()))}")
    print(f"dead episodes: base {bdead}, trained {tdead} | endings base {dict(bends)} trained {dict(tends)}")
    rng = random.Random(300)
    idx = [[rng.randrange(len(tasks)) for _ in tasks] for _ in range(B)]
    for label, col in [("full reward", 0), ("DB check only", 1)]:
        print(f"\n== {label}")
        for k in (1, 2, 4):
            # A task with fewer than k scored trials in either arm has no pass^k; leave it out.
            ok = [min(len(base[x]), len(tr[x])) >= k for x in tasks]
            b = [pass_k([t[col] for t in base[x]], k) if o else 0.0 for x, o in zip(tasks, ok)]
            t = [pass_k([t[col] for t in tr[x]], k) if o else 0.0 for x, o in zip(tasks, ok)]
            d = [y - x for x, y in zip(b, t)]
            mb, mt, md = (100 * sum(v) / sum(ok) for v in (b, t, d))
            bs = sorted(100 * sum(d[i] for i in ix) / max(sum(ok[i] for i in ix), 1) for ix in idx)
            lo, hi = bs[int(0.025 * B)], bs[int(0.975 * B)]
            p_le0 = sum(v <= 0 for v in bs) / B
            note = "" if all(ok) else f"  ({sum(ok)} tasks)"
            print(f"pass^{k}: base {mb:5.1f}  trained {mt:5.1f}  diff {md:+5.1f}  95% CI [{lo:+5.1f}, {hi:+5.1f}]  P(diff<=0) {p_le0:.3f}{note}")
    cb = {x: sum(t[0] for t in base[x]) for x in tasks}
    ct = {x: sum(t[0] for t in tr[x]) for x in tasks}
    print(f"\nper-task success count: up {sum(ct[x] > cb[x] for x in tasks)}, down {sum(ct[x] < cb[x] for x in tasks)}, same {sum(ct[x] == cb[x] for x in tasks)}")
    print("histogram c/8   base:", [sum(cb[x] == c for x in tasks) for c in range(9)])
    print("histogram c/8 trained:", [sum(ct[x] == c for x in tasks) for c in range(9)])


main()
