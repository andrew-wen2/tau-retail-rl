"""Tasks the 2B never solved: pooled over the latest profile plus every training/profiling attempt on record."""
import json, glob, collections
from records import iter_episodes
prof = json.load(open("tasks/profile-v9.json"))
sp = json.load(open("tasks/split_v9.json"))
att = collections.defaultdict(lambda: [0, 0])
for d in glob.glob("runs/p4-[fghi]/records/b*") + glob.glob("runs/profile*/**/b*", recursive=True):
    for e in iter_episodes(d):
        if not e.needs_reroll: att[e.task_id][0] += e.eval_reward; att[e.task_id][1] += 1
train = set(sp["train"])
rows = []
for t in train:
    s, n = prof.get(t, {}).get("successes", 0), prof.get(t, {}).get("trials", 0)
    s2, n2 = att.get(t, [0, 0])
    rows.append((t, max(s, s2), max(n, n2)))
fam = lambda t: t.split("_")[0] + ("_" + t.split("_")[1] if t.startswith("t6") else "")
for lo in (4, 8, 16):
    z = [r for r in rows if r[1] == 0 and r[2] >= lo]
    print(f"train tasks with 0 successes in >= {lo} attempts: {len(z)} / {len(rows)}", dict(collections.Counter(fam(r[0]) for r in z)))
lowp = [r for r in rows if r[2] >= 4 and r[1] / r[2] < 0.5]
print("train tasks with p < 0.5 (>=4 attempts):", len(lowp))
json.dump(sorted(r[0] for r in rows if r[1] == 0 and r[2] >= 4), open("zero_tasks.json", "w"))
json.dump({r[0]: [r[1], r[2]] for r in rows}, open("task_success.json", "w"))
