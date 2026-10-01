import sys; sys.path.insert(0, ".")  # run from the repo root as experiments/<script>
"""Re-score past valbig reads with tau3's rule (DB x COMMUNICATE) instead of DB only."""
import collections, json, random
from driver import communicated
from records import AssistantTurn, iter_episodes
v5 = {t["id"]: t for t in json.load(open("tasks/tb_train_v5.json"))}
def infos(tid): return v5[tid]["evaluation_criteria"].get("communicate_info") or []
def load(d, strict):
    m = collections.defaultdict(list); unsaid = 0; n = 0
    for e in iter_episodes(d):
        if e.needs_reroll: continue
        r = e.eval_reward
        if not strict and e.termination == "natural" and e.extra.get("db_ok") is not None:
            r = int(e.extra["db_ok"])  # reads made after Sept 28 already carry the tau3 rule
        if strict and r:
            ok = communicated([t.text for t in e.turns if isinstance(t, AssistantTurn)], infos(e.task_id))
            unsaid += not ok; r = int(ok)
        m[e.task_id].append(r); n += 1
    return m, unsaid / max(n, 1)
from math import comb
pk = lambda xs, k: comb(sum(xs), k) / comb(len(xs), k)
def cmp(name, a, b):
    ts = sorted(t for t in b if t in a and len(a[t]) >= 4 and len(b[t]) >= 4)
    out = [f"{name:<20}"]
    for k in (1, 4):
        d = [pk(b[t], k) - pk(a[t], k) for t in ts]; random.seed(k)
        bs = sorted(sum(random.choices(d, k=len(d))) / len(d) for _ in range(4000))
        ma = sum(pk(a[t], k) for t in ts) / len(ts)
        out.append(f"p^{k} {ma:.3f}->{ma + sum(d)/len(d):.3f} {sum(d)/len(d):+.3f} [{bs[100]:+.3f},{bs[3899]:+.3f}]")
    print(" | ".join(out))
arms = [("base", "runs/valbig/base"), ("p4-b s15", "runs/p4-e/val/v0000"), ("p4-f s8", "runs/valbig/p4f-008"),
        ("p4-g 006", "runs/valbig/p4g-006"), ("p4-h s4", "runs/p4-h/val/v0004"), ("p4-i s4", "runs/p4-i/val/v0004"),
        ("soup4", "runs/valbig/soup4"), ("sft-a", "runs/valbig/sft-a-2")]
import os
for n in ("soupj", "p4k", "soupk"):
    if os.path.isdir(f"runs/valbig/{n}") and list(__import__("pathlib").Path(f"runs/valbig/{n}").glob("*.json.gz")):
        arms.append((n, f"runs/valbig/{n}"))
S = {}
print(f"{'arm':<10} {'DB-only p^1':>11} {'tau3-rule p^1':>13} {'right DB but unsaid':>20}")
for n, d in arms:
    lo, _ = load(d, False); st, un = load(d, True); S[n] = st
    f = lambda m: sum(map(sum, m.values())) / sum(map(len, m.values()))
    print(f"{n:<10} {f(lo):>11.3f} {f(st):>13.3f} {un:>20.3f}")
print("== paired vs base, tau3 rule")
for n, _ in arms[1:]: cmp(n, S["base"], S[n])
print("== vs soup4, tau3 rule")
cmp("sft-a", S["soup4"], S["sft-a"]); cmp("p4-f s8", S["soup4"], S["p4-f s8"])
for n in ("soupj", "p4k", "soupk"):
    if n in S: cmp(n, S["soup4"], S[n])
