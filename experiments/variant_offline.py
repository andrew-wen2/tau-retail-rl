"""Offline (Sept 29): is variant-closeness credit for wrong item ids worth adding?

For every failed training episode in p4-j (and p4-k so far), each gold swap pair (old -> gold
new) is compared with what the agent actually swapped that old item to, on the same order:
  exact | wrong variant of the same product | not swapped.
For wrong variants: option closeness to the gold variant vs the closeness a random other
variant of the product would get (chance), and whether the agent's chosen id appeared in any
tool result earlier in the episode (chose among real options vs invented / mis-copied).
Then: how the extra credit would change group-level learning signal."""
import sys; sys.path.insert(0, ".")
import collections, glob, json, statistics
from records import (READ_ONLY_TOOLS, AssistantTurn, EnvTurn, _successful_writes,
                     graded_write_credit, iter_episodes)
SWAPS = ("modify_pending_order_items", "exchange_delivered_order_items")
db = json.load(open("baseline/tau2-bench/data/tau2/domains/retail/db.json"))
item = {}
for p in db["products"].values():
    for iid, v in p["variants"].items():
        item[iid] = (p["product_id"], v["options"])
variants = collections.defaultdict(list)
for iid, (pid, _) in item.items():
    variants[pid].append(iid)
tasks = {t["id"]: t for t in json.load(open("tasks/tb_train_v10.json"))}
def close(a, b):
    oa, ob = item[a][1], item[b][1]
    keys = set(oa) | set(ob)
    return sum(oa.get(k) == ob.get(k) for k in keys) / len(keys)
def chance(gold_new, exclude):
    pid = item[gold_new][0]
    others = [i for i in variants[pid] if i not in (gold_new, exclude)]
    return statistics.mean(close(o, gold_new) for o in others) if others else 0.0
def seen_before(ep, iid, upto_turn):
    for t in ep.turns[:upto_turn]:
        if isinstance(t, EnvTurn) and t.role == "tool" and iid in t.content:
            return True
    return False
cls = collections.Counter(); closes = []; chances = []; seen = collections.Counter()
per_ep_bonus = {}
groups = collections.defaultdict(list)
dirs = sorted(glob.glob("runs/p4-j*/records/b*")) + sorted(glob.glob("runs/p4-k/records/b*"))
n_fail = n_wrongvar_eps = 0
for d in dirs:
    for e in iter_episodes(d):
        if e.needs_reroll or not e.trains:
            continue
        gold = tasks[e.task_id]["evaluation_criteria"]["actions"]
        base = 0.3 * graded_write_credit(e, gold) if not e.eval_reward else 1.0
        bonus = 0.0
        if not e.eval_reward:
            n_fail += 1
            # the agent's successful swaps with the turn index they happened at
            done = []
            for i, t in enumerate(e.turns):
                if isinstance(t, AssistantTurn):
                    for c in t.tool_calls or []:
                        if c.get("name") in SWAPS:
                            done.append((i, c["name"], c.get("arguments") or {}))
            ok = {(n, json.dumps(a, sort_keys=True)) for n, a in _successful_writes(e) if n in SWAPS}
            done = [(i, n, a) for i, n, a in done if (n, json.dumps(a, sort_keys=True)) in ok]
            gpairs = [(g["name"], g["arguments"]["order_id"], o, nw) for g in gold if g["name"] in SWAPS
                      for o, nw in zip(g["arguments"]["item_ids"], g["arguments"]["new_item_ids"])]
            wrongvar = False; add = []
            for name, oid, old, gnew in gpairs:
                agent = [(i, dict(zip(a.get("item_ids", []), a.get("new_item_ids", []))).get(old))
                         for i, n, a in done if n == name and a.get("order_id") == oid]
                agent = [(i, x) for i, x in agent if x]
                if not agent:
                    cls["not swapped"] += 1; continue
                ti, anew = agent[0]
                if anew == gnew:
                    cls["exact"] += 1; continue
                if anew not in item or item[anew][0] != item[gnew][0]:
                    cls["other product / unknown id"] += 1; continue
                cls["wrong variant, same product"] += 1; wrongvar = True
                c, ch = close(anew, gnew), chance(gnew, anew)
                closes.append(c); chances.append(ch)
                seen["chosen id seen in an earlier tool result" if seen_before(e, anew, ti) else "chosen id never shown (invented / mis-copied)"] += 1
                add.append(max(0.0, (c - ch) / (1 - ch)) if ch < 1 else 0.0)
            n_wrongvar_eps += wrongvar
            if add and gpairs:
                bonus = 0.3 * sum(add) / len(gpairs)
        groups[e.group_id].append((e.eval_reward, base, min(0.3, base + bonus) if not e.eval_reward else 1.0))
print(f"failed episodes {n_fail} | with >=1 wrong-variant swap on the right item {n_wrongvar_eps} ({n_wrongvar_eps/max(1,n_fail):.0%})")
print("gold swap pairs in failed episodes:", dict(cls))
if closes:
    print(f"wrong-variant closeness to gold: mean {statistics.mean(closes):.2f} vs chance {statistics.mean(chances):.2f} "
          f"| above chance in {sum(c > h for c, h in zip(closes, chances))}/{len(closes)} | exactly 1 option off "
          f"{sum(c >= 0.74 for c in closes)}")
print("chosen wrong ids:", dict(seen))
def sig(v, i): return statistics.pstdev([x[i] for x in v]) > 1e-6
mixed_changed = 0
for v in groups.values():
    f = [x for x in v if not x[0]]
    if len(f) > 1 and statistics.pstdev([x[1] for x in f]) < 1e-6 and statistics.pstdev([x[2] for x in f]) > 1e-6:
        mixed_changed += 1
print(f"groups {len(groups)} | with signal: graded {sum(sig(v,1) for v in groups.values())}, graded+variant {sum(sig(v,2) for v in groups.values())} "
      f"| groups whose failures become distinguishable only with the variant term: {mixed_changed}")
print(f"mean failure reward: graded {statistics.mean(x[1] for v in groups.values() for x in v if not x[0]):.3f}, "
      f"with variant {statistics.mean(x[2] for v in groups.values() for x in v if not x[0]):.3f}")
