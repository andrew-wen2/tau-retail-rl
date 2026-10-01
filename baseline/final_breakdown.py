"""Exploratory breakdown of the final eval: where the trained arm gained and lost against the base arm.

Three reads, all from the two results.json files (no API calls):
  1. pass^1 difference by tau3 task type (paired bootstrap over the tasks in each group).
     Post hoc and 16-way: treat any single group as a hypothesis, not a finding.
  2. On tasks whose gold actions contain no write: failures, and whether the agent wrote anyway.
  3. Authentication on the eval trajectories: did an authentication call precede the first write?

Usage: python final_breakdown.py BASE_RESULTS.json TRAINED_RESULTS.json"""
import json, random, sys
from collections import Counter

READ = {"find_user_id_by_email", "find_user_id_by_name_zip", "get_user_details", "get_order_details",
        "get_product_details", "get_item_details", "list_all_product_types", "calculate",
        "transfer_to_human_agents", "think"}
AUTH = {"find_user_id_by_email", "find_user_id_by_name_zip", "get_user_details"}
WRITE_TOOLS = ["return_delivered_order_items", "exchange_delivered_order_items", "cancel_pending_order",
               "modify_pending_order_items", "modify_pending_order_address", "modify_user_address"]


def gold_writes(task):
    return [a for a in (task.get("evaluation_criteria") or {}).get("actions") or [] if a["name"] not in READ]


def calls(sim):
    return [tc["name"] for m in sim["messages"] if m["role"] == "assistant" for tc in (m.get("tool_calls") or [])]


def success_counts(d):
    c = Counter()
    for s in d["simulations"]:
        if s.get("reward_info") and s["reward_info"]["reward"] >= 1:
            c[s["task_id"]] += 1
    return c


def main():
    B, T = (json.load(open(p)) for p in sys.argv[1:3])
    tasks = {t["id"]: t for t in B["tasks"]}
    cb, ct = success_counts(B), success_counts(T)
    nb, nt = (Counter(s["task_id"] for s in d["simulations"]) for d in (B, T))
    rows = []
    for tid, t in tasks.items():
        ec = t.get("evaluation_criteria") or {}
        w = gold_writes(t)
        rows.append(dict(nl=bool(ec.get("nl_assertions")), comm=bool(ec.get("communicate_info")),
                         nw=min(len(w), 3), tools={a["name"] for a in w},
                         orders=len({json.dumps(a.get("arguments", {}).get("order_id")) for a in w}),
                         d=ct[tid] / nt[tid] - cb[tid] / nb[tid], base=cb[tid] / nb[tid]))
    rng = random.Random(0)
    print("== 1. pass^1 difference by task type (points, trained - base, 95% CI)")
    groups = [("all", lambda r: True), ("has NL assertions", lambda r: r["nl"]),
              ("no NL assertions", lambda r: not r["nl"]), ("has communicate_info", lambda r: r["comm"]),
              ("0 writes", lambda r: r["nw"] == 0), ("1 write", lambda r: r["nw"] == 1),
              ("2 writes", lambda r: r["nw"] == 2), ("3+ writes", lambda r: r["nw"] == 3),
              ("writes on 2+ orders", lambda r: r["orders"] >= 2)]
    groups += [(f"uses {t}", lambda r, t=t: t in r["tools"]) for t in WRITE_TOOLS]
    for name, pred in groups:
        sel = [r for r in rows if pred(r)]
        d = [r["d"] for r in sel]
        bs = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(4000))
        print(f"{name:36s} n={len(sel):3d}  base {sum(r['base'] for r in sel) / len(sel):.2f}  "
              f"diff {100 * sum(d) / len(d):+5.1f} [{100 * bs[100]:+5.1f}, {100 * bs[3900]:+5.1f}]  "
              f"up/down {sum(x > 0 for x in d)}/{sum(x < 0 for x in d)}")

    print("\n== 2. tasks whose gold has no write")
    zero = {tid for tid, t in tasks.items() if not gold_writes(t)}
    for name, d in (("base", B), ("trained", T)):
        sims = [s for s in d["simulations"] if s["task_id"] in zero]
        fails = [s for s in sims if s["reward_info"]["reward"] < 1]
        wrote = [s for s in fails if any(c not in READ for c in calls(s))]
        what = Counter(c for s in wrote for c in calls(s) if c not in READ)
        print(f"{name:8s} {len(zero)} tasks, {len(sims)} episodes: {len(fails)} failures, "
              f"{len(wrote)} of them made a write: {dict(what)}")

    print("\n== 3. authentication before the first write (episodes that wrote)")
    for name, d in (("base", B), ("trained", T)):
        ok = tot = 0
        for s in d["simulations"]:
            cs = calls(s)
            first = next((i for i, c in enumerate(cs) if c not in READ), None)
            if first is None:
                continue
            tot += 1
            ok += any(c in AUTH for c in cs[:first])
        print(f"{name:8s} {ok}/{tot} = {ok / tot:.3f}")


main()
