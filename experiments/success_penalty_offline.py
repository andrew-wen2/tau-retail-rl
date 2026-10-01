"""Offline (Sept 29): in p4-j's groups, how often do SUCCESSFUL episodes contain write mistakes
-- rejected write calls, successful writes matching no gold write (wrong-then-fixed), writes to
untouched targets -- and how many all-success groups would gain signal if successes carrying
them were docked."""
import sys; sys.path.insert(0, ".")
import collections, glob, json, statistics
from records import (READ_ONLY_TOOLS, AssistantTurn, EnvTurn, gold_write_progress,
                     iter_episodes, unrequested_writes)
tasks = {t["id"]: t for t in json.load(open("tasks/tb_train_v10.json"))}
def rejected_writes(ep):
    n = 0
    for i, t in enumerate(ep.turns):
        if not isinstance(t, AssistantTurn): continue
        calls = t.tool_calls or ([t.tool_call] if t.tool_call else [])
        res = []
        for r in ep.turns[i + 1:]:
            if isinstance(r, EnvTurn) and r.role == "tool": res.append(r)
            else: break
        for j, c in enumerate(calls):
            if c.get("name") and c["name"] not in READ_ONLY_TOOLS and j < len(res) and res[j].content.startswith("Error"):
                n += 1
    return n
groups = collections.defaultdict(list)
for d in sorted(glob.glob("runs/p4-j*/records/b*")):
    for e in iter_episodes(d):
        if e.needs_reroll or not e.trains: continue
        gold = tasks[e.task_id]["evaluation_criteria"]["actions"]
        _, _, extra = gold_write_progress(e, gold)
        groups[e.group_id].append(dict(ok=e.eval_reward, rej=rejected_writes(e), extra=extra,
                                      unreq=unrequested_writes(e, gold), turns=e.turn_count))
succ = [x for v in groups.values() for x in v if x["ok"]]
print(f"successful episodes {len(succ)}: with a rejected write {sum(x['rej']>0 for x in succ)} ({sum(x['rej']>0 for x in succ)/len(succ):.1%}), "
      f"with a non-gold successful write (wrong-then-fixed) {sum(x['extra']>0 for x in succ)} ({sum(x['extra']>0 for x in succ)/len(succ):.1%}), "
      f"with an unrequested write {sum(x['unreq']>0 for x in succ)} ({sum(x['unreq']>0 for x in succ)/len(succ):.1%})")
alls = [v for v in groups.values() if all(x["ok"] for x in v)]
flag = lambda x: x["rej"] > 0 or x["extra"] > 0 or x["unreq"] > 0
gain = [v for v in alls if 0 < sum(map(flag, v)) < len(v)]
print(f"all-success groups {len(alls)} of {len(groups)}: would gain signal from a mistake penalty {len(gain)} "
      f"(flagged episodes per such group: {[sum(map(flag, v)) for v in gain][:20]})")
# are stumbling successes less reliable? task-level: success rate of the task vs share of its successes that stumbled
by_task = collections.defaultdict(list)
for gid, v in groups.items(): by_task[gid.split("-", 2)[2]].extend(v)
rows = [(sum(x["ok"] for x in v)/len(v), sum(flag(x) for x in v if x["ok"])/max(1, sum(x["ok"] for x in v))) for v in by_task.values() if sum(x["ok"] for x in v)]
lo = [s for p, s in rows if p < 0.75]; hi = [s for p, s in rows if p >= 0.75]
print(f"share of successes that stumbled: on tasks solved <75% of the time {statistics.mean(lo):.1%} (n={len(lo)}), >=75% {statistics.mean(hi):.1%} (n={len(hi)})")
print("median turns: clean successes", statistics.median(x["turns"] for x in succ if not flag(x)), "| stumbling successes", statistics.median(x["turns"] for x in succ if flag(x)) if any(map(flag, succ)) else None)
