"""Offline: what graded write credit would have done to p4-j's learning signal (Sept 29).
Re-scores every recorded p4-j episode with the gated and the graded training reward (same
partial_credit 0.3 and unrequested penalty 0.5) and compares group-level signal."""
import sys; sys.path.insert(0, ".")
import collections, glob, json, statistics
from records import gold_write_progress, graded_write_credit, iter_episodes, unrequested_writes
tasks = {t["id"]: t for t in json.load(open("tasks/tb_train_v10.json"))}
C, U = 0.3, 0.5
def rewards(e, gold):
    if e.eval_reward:
        return 1.0, 1.0
    m, t, w = gold_write_progress(e, gold)
    gated = C * m / t if t and not w else 0.0
    graded = C * graded_write_credit(e, gold)
    pen = U * (unrequested_writes(e, gold) > 0)
    return gated - pen, graded - pen
groups = collections.defaultdict(list)
for d in sorted(glob.glob("runs/p4-j*/records/b*")):
    for e in iter_episodes(d):
        if e.needs_reroll or not e.trains:
            continue
        gold = tasks[e.task_id]["evaluation_criteria"]["actions"]
        groups[e.group_id].append((e.eval_reward, *rewards(e, gold)))
fails = [(g1, g2) for v in groups.values() for r, g1, g2 in v if r == 0]
print(f"groups {len(groups)} | failed episodes {len(fails)}")
print(f"failed episodes with credit > 0: gated {sum(g > 0 for g, _ in fails)} ({sum(g > 0 for g, _ in fails)/len(fails):.0%}), "
      f"graded {sum(g > 0 for _, g in fails)} ({sum(g > 0 for _, g in fails)/len(fails):.0%})")
print(f"mean training reward of failures: gated {statistics.mean(g for g, _ in fails):+.3f}, graded {statistics.mean(g for _, g in fails):+.3f}")
def sig(v, i):
    xs = [x[i] for x in v]
    return statistics.pstdev(xs) > 1e-6
allfail = [v for v in groups.values() if not any(x[0] for x in v)]
allsucc = [v for v in groups.values() if all(x[0] for x in v)]
mixed = [v for v in groups.values() if v not in allfail and v not in allsucc]
print(f"all-fail groups {len(allfail)}: with any signal -- gated {sum(sig(v,1) for v in allfail)}, graded {sum(sig(v,2) for v in allfail)}")
print(f"mixed groups {len(mixed)} | all-success {len(allsucc)} (no signal under either)")
print(f"groups with signal overall: gated {sum(sig(v,1) for v in groups.values())}, graded {sum(sig(v,2) for v in groups.values())} of {len(groups)}")
# in mixed groups: does any failure now score close to a success? (credit is capped at 0.3)
print(f"max graded credit on a failure: {max(g for _, g in fails):.3f}")
