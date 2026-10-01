"""Paired gpt-5.2 proxy comparison under tau3's rule (DB x COMMUNICATE) for every arm; the Sept 28
arms were recorded DB-only, so all are re-scored the same way here."""
import sys; sys.path.insert(0, ".")
import collections, json, random
from driver import communicated
from records import AssistantTurn, iter_episodes
v5 = {t["id"]: t for t in json.load(open("tasks/tb_train_v5.json"))}
def load(d):
    m = collections.defaultdict(list)
    for e in iter_episodes(d):
        if e.needs_reroll: continue
        r = e.eval_reward
        if e.termination == "natural" and e.extra.get("db_ok") is not None:
            r = int(e.extra["db_ok"])
        if r:
            r = int(communicated([t.text for t in e.turns if isinstance(t, AssistantTurn)],
                                 v5[e.task_id]["evaluation_criteria"].get("communicate_info") or []))
        m[e.task_id].append(r)
    return m
def cmp(label, a, b):
    ts = sorted(t for t in a if t in b and len(a[t]) == len(b[t]) >= 2)
    d = [sum(b[t]) / len(b[t]) - sum(a[t]) / len(a[t]) for t in ts]
    random.seed(0); bs = sorted(sum(random.choices(d, k=len(d))) / len(d) for _ in range(4000))
    ma = sum(sum(a[t]) / len(a[t]) for t in ts) / len(ts)
    print(f"{label:<28} n={len(ts)} pass^1 {ma:.3f} -> {ma + sum(d)/len(d):.3f}  diff {sum(d)/len(d):+.3f} "
          f"[{bs[100]:+.3f}, {bs[3899]:+.3f}]  up {sum(x > 0 for x in d)} down {sum(x < 0 for x in d)}")
A = {n: load(f"runs/proxy52/{n}") for n in ("base", "soup4", "soupk", "soupm")}
print("gpt-5.2 customer, 94 valbig tasks x 2, tau3 rule")
cmp("base -> soup4", A["base"], A["soup4"])
cmp("base -> soupk", A["base"], A["soupk"])
cmp("soup4 -> soupk", A["soup4"], A["soupk"])
cmp("base -> soupm", A["base"], A["soupm"])
cmp("soup4 -> soupm", A["soup4"], A["soupm"])
