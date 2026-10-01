"""Build the v5 training set (Sept 26, run p4-e). Run on the VM in .venv-tau2 (needs tau2).

  1. clean v4's train tasks of the task noise the p4-b transcript read found:
       - drop tasks whose request renders an empty change ("... to {}")
       - forbidden-first returns said "cancel this order entirely" before any order was named,
         and the customer picked an arbitrary pending order: name it as the order below
       - tell every training customer never to state an order/item id it was not given
         (customers invented id-to-item mappings the agent then acted on)
  2. add gen_targeted.py's tasks after: schema check, gold replay, a DB change, no write
     signature shared with the held-out 114 or with any val task, and -- for multi-item swaps --
     the same end state when the item lists are given in reverse order. tau2's
     modify_pending_order_items gives every modified item the LAST new item's price and
     options (the leaked `variant`, tools.py ~533), so for those the DB hash depends on list
     order and an agent listing the items differently from the gold fails for nothing.
  3. val = tb_valbig.json (the 60 val tasks + 106 variants, 166).

Writes tb_train_v5.json and split_v5.json.
"""

from __future__ import annotations

import json
from collections import Counter

from tau2.data_model.tasks import Task
from tau2.domains.retail.environment import get_environment, get_tasks

READ = {"calculate", "find_user_id_by_email", "find_user_id_by_name_zip", "get_item_details",
        "get_order_details", "get_product_details", "get_user_details", "list_all_product_types",
        "think", "transfer_to_human_agents"}
HONEST = (" Never state an order id or item id you were not given; refer to your orders by the "
          "items in them.")
FORBID_OLD = "Start by asking the agent to cancel this order entirely."
FORBID_NEW = ("Start by asking the agent to cancel the order described below entirely (the one "
              "you want to return items from).")


def sig(t: dict) -> set[tuple]:
    return {(a["name"], a["arguments"].get("order_id", a["arguments"].get("user_id")))
            for a in t["evaluation_criteria"]["actions"] if a["name"] not in READ}


def replay(actions: list[dict]) -> str:
    env = get_environment()
    for a in actions:
        env.make_tool_call(tool_name=a["name"], requestor=a.get("requestor", "assistant"),
                           **a["arguments"])
    return env.get_db_hash()


def reversed_lists(actions: list[dict]) -> list[dict]:
    out = json.loads(json.dumps(actions))
    for a in out:
        if "new_item_ids" in a["arguments"] and len(a["arguments"]["item_ids"]) > 1:
            a["arguments"]["item_ids"] = a["arguments"]["item_ids"][::-1]
            a["arguments"]["new_item_ids"] = a["arguments"]["new_item_ids"][::-1]
    return out


def main() -> int:
    v4 = {t["id"]: t for t in json.load(open("tasks/tb_train_v4.json"))}
    sp4 = json.load(open("tasks/split_v4.json"))
    valbig = json.load(open("tasks/tb_valbig.json"))
    held = set().union(*(sig(x.model_dump()) for x in get_tasks()))
    val_sig = set().union(*(sig(t) for t in valbig))
    base_hash = get_environment().get_db_hash()

    # 1. clean
    train, why = [], Counter()
    for tid in sp4["train"]:
        t = json.loads(json.dumps(v4[tid]))
        ins = t["user_scenario"]["instructions"]
        if "to {}" in ins["reason_for_call"]:
            why["dropped: empty change 'to {}'"] += 1
            continue
        if FORBID_OLD in ins["reason_for_call"]:
            ins["reason_for_call"] = ins["reason_for_call"].replace(FORBID_OLD, FORBID_NEW)
            why["forbidden-first reworded"] += 1
        ins["task_instructions"] = (ins.get("task_instructions") or "") + HONEST
        train.append(t)

    # 2. targeted
    tt = json.load(open("tasks/tb_targeted.json"))
    keep, drop = [], Counter()
    for t in tt:
        try:
            Task.model_validate(t)
        except Exception:  # noqa: BLE001
            drop["schema"] += 1
            continue
        try:
            h = replay(t["evaluation_criteria"]["actions"])
        except Exception as e:  # noqa: BLE001
            drop[f"replay: {str(e)[:50]}"] += 1
            continue
        if h == base_hash:
            drop["no DB change"] += 1
            continue
        if sig(t) & held:
            drop["shares a write with the 114"] += 1
            continue
        if sig(t) & val_sig:
            drop["shares a write with val"] += 1
            continue
        acts = t["evaluation_criteria"]["actions"]
        if any("new_item_ids" in a["arguments"] and len(a["arguments"]["item_ids"]) > 1 for a in acts):
            try:
                if replay(reversed_lists(acts)) != h:
                    drop["end state depends on item order (tau2 bug)"] += 1
                    continue
            except Exception:  # noqa: BLE001
                drop["reversed replay errored"] += 1
                continue
        keep.append(t)

    tasks = train + keep + valbig
    ids = [t["id"] for t in tasks]
    assert len(ids) == len(set(ids)), "duplicate ids"
    split = {"train": sorted(t["id"] for t in train + keep), "val": sorted(t["id"] for t in valbig)}
    split["all"] = sorted(split["train"] + split["val"])
    assert not set(split["train"]) & set(split["val"])
    json.dump(tasks, open("tasks/tb_train_v5.json", "w"), indent=1)
    json.dump(split, open("tasks/split_v5.json", "w"), indent=1)
    fam = Counter(t["id"][3:].rsplit("_", 1)[0] for t in keep)
    print(f"v5: train {len(split['train'])} = cleaned v4 {len(train)} + targeted {len(keep)} | "
          f"val {len(split['val'])}")
    print("cleaning:", dict(why))
    print("targeted dropped:", dict(drop))
    print("targeted kept by family/level:", dict(sorted(fam.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
