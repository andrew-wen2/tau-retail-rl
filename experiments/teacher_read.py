import json, collections
from records import iter_episodes
d = json.load(open("task_success.json"))
zero = set(json.load(open("zero_tasks.json")))
t = collections.defaultdict(list); term = collections.Counter()
for e in iter_episodes("runs/teacher/v41f"):
    term[e.termination] += 1
    if not e.needs_reroll: t[e.task_id].append(e.eval_reward)
print("episodes", sum(map(len, t.values())), "tasks", len(t), "terminations", dict(term))
def band(p): return "0" if p == 0 else "(0,.25)" if p < .25 else "[.25,.5)"
rows = collections.defaultdict(lambda: [0, 0, 0])
for k, v in t.items():
    s, n = d[k]; r = rows[band(s / n)]
    r[0] += 1; r[1] += sum(v) / len(v); r[2] += sum(v) > 0
for k in ("0", "(0,.25)", "[.25,.5)"):
    n, m, a = rows[k]
    print(f"2B success {k:<9} tasks {n:3d} | DeepSeek-4.1 pass^1 {m/n:.2f} | solved at least once {a}")
unsolv = sorted(k for k in zero if k in t and sum(t[k]) == 0)
print("never-solved tasks DeepSeek also failed (delete):", len(unsolv), "of", len([k for k in zero if k in t]))
json.dump(unsolv, open("tasks/unsolvable_tasks.json", "w"))
print("fine-tune candidate tasks (DeepSeek solved >= once):", sum(sum(v) > 0 for v in t.values()),
      "| successful transcripts:", sum(sum(v) for v in t.values()))
