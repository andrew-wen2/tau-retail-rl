"""Dump 20 wrong-variant failures (p4-j) compactly: the task's instruction, the gold vs chosen
variant options (with the old item's), what the customer said, and the agent's text around
the swap. Random sample, fixed seed."""
import sys; sys.path.insert(0, ".")
import collections, glob, json, random
from records import AssistantTurn, EnvTurn, _successful_writes, iter_episodes
SWAPS = ("modify_pending_order_items", "exchange_delivered_order_items")
db = json.load(open("baseline/tau2-bench/data/tau2/domains/retail/db.json"))
item = {iid: (p["name"], v["options"], v["available"], v["price"]) for p in db["products"].values() for iid, v in p["variants"].items()}
tasks = {t["id"]: t for t in json.load(open("tasks/tb_train_v10.json"))}
cases = []
for d in sorted(glob.glob("runs/p4-j*/records/b*")):
    for e in iter_episodes(d):
        if e.needs_reroll or e.eval_reward or not e.trains: continue
        gold = tasks[e.task_id]["evaluation_criteria"]["actions"]
        gp = {(g["arguments"]["order_id"], o): n for g in gold if g["name"] in SWAPS
              for o, n in zip(g["arguments"]["item_ids"], g["arguments"]["new_item_ids"])}
        for n, a in _successful_writes(e):
            if n not in SWAPS: continue
            for o, nw in zip(a.get("item_ids", []), a.get("new_item_ids", [])):
                g = gp.get((a.get("order_id"), o))
                if g and g != nw and nw in item and item[nw][0] == item[g][0]:
                    cases.append((e, o, g, nw))
random.seed(29)
pick = random.sample(cases, min(20, len(cases)))
for k, (e, old, g, nw) in enumerate(pick):
    t = tasks[e.task_id]
    print(f"\n######## {k+1}. {e.task_id} persona={e.extra.get('persona')} ({e.episode_id})")
    us = t["user_scenario"]["instructions"]
    print("INSTRUCTION:", (us if isinstance(us, str) else json.dumps(us))[:900])
    print(f"PRODUCT {item[g][0]} | old {item[old][1]}\n  gold   {item[g][1]} avail={item[g][2]} ${item[g][3]}\n  chosen {item[nw][1]} avail={item[nw][2]} ${item[nw][3]}")
    # customer messages mentioning the product, and the agent turn that proposed the swap
    shown = 0
    for i, turn in enumerate(e.turns):
        if isinstance(turn, EnvTurn) and turn.role == "user" and shown < 4:
            print("  CUSTOMER:", turn.content[:350].replace("\n", " ")); shown += 1
    for i, turn in enumerate(e.turns):
        if isinstance(turn, AssistantTurn) and nw in (turn.text or "") :
            print("  AGENT (mentions chosen id):", turn.text[:700].replace("\n", " ")); break
    for i, turn in enumerate(e.turns):
        if isinstance(turn, AssistantTurn) and any(c.get("name") in SWAPS for c in (turn.tool_calls or [])):
            prev = [x for x in e.turns[max(0, i-3):i] if isinstance(x, EnvTurn) and x.role == "user"]
            if prev: print("  CUSTOMER just before swap:", prev[-1].content[:300].replace("\n", " "))
            break
print(f"\n(total wrong-variant cases: {len(cases)})")
