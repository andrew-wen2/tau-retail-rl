"""Per-persona breakdown of one training batch: python experiments/persona_step.py <run> <batch>."""
import sys; sys.path.insert(0, ".")
import collections, json, statistics
from records import EnvTurn, failure_class, iter_episodes, rejected_writes
run, b = sys.argv[1], sys.argv[2]
tasks = {t["id"]: t for t in json.load(open("tasks/tb_train_v11.json"))}
prof = json.load(open("tasks/profile-v11.json"))
eps = [e for e in iter_episodes(f"runs/{run}/records/{b}") if not e.needs_reroll]
by = collections.defaultdict(list)
for e in eps:
    by[e.extra.get("persona")].append(e)
print(f"{'persona':<12}{'groups':>7}{'eps':>5}{'reward':>8}{'prior':>7}{'diff':>7}{'all-succ':>9}{'all-fail':>9}{'mixed':>6}{'stumbled':>9}{'open':>6}")
rest = []
for p in ("forthcoming", "terse", "impatient", "uncertain"):
    v = by.get(p, [])
    if not v:
        print(f"{p:<12} none"); continue
    g = collections.defaultdict(list)
    for e in v:
        g[e.group_id].append(e.eval_reward)
    r = sum(e.eval_reward for e in v) / len(v)
    pr = [prof[t]["successes"] / prof[t]["trials"] for t in {e.task_id for e in v} if t in prof]
    pr = sum(pr) / len(pr) if pr else float("nan")
    succ = [e for e in v if e.eval_reward]
    fc = collections.Counter(failure_class(e, tasks[e.task_id]["evaluation_criteria"]["actions"]) for e in v if not e.eval_reward)
    op = statistics.median(len(next(t.content for t in e.turns if isinstance(t, EnvTurn) and t.role == "user")) for e in v)
    per = {gid.split("-", 2)[2]: f"{sum(x)}/{len(x)}" for gid, x in g.items()}
    print(f"{p:<12}{len(g):>7}{len(v):>5}{r:>8.3f}{pr:>7.3f}{r - pr:>+7.3f}{sum(all(x) for x in g.values()):>9}"
          f"{sum(not any(x) for x in g.values()):>9}{sum(0 < sum(x) < len(x) for x in g.values()):>6}"
          f"{sum(rejected_writes(e) > 0 for e in succ) / max(1, len(succ)):>9.2f}{op:>6.0f}")
    rest.append(f"  {p}: failures {dict(fc)} | per task {per}")
print("\n".join(rest))
