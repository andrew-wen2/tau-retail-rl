"""Build the v4 training set (Sept 26): the hardened 462, their re-rolled train variants
(vary_tb.py -> tb_vary.json) and the 150 tau2-gen tasks (tb_gen.json, generated on the VM
Sept 25). Val is unchanged -- the same 60 hardened val tasks -- so v4 runs compare directly
with p4-b and p4-c.

Leak check against the held-out tau3 114 by write signature: (tool, order id) for order
tools and (tool, user id) for modify_user_address, which has no order id (the Sept 25 check
keyed it on order id and so matched every address change to every other). Run on the VM in
.venv-tau2 (needs tau2 for the 114). Writes tb_train_v4.json and split_v4.json.
"""

from __future__ import annotations

import json
from pathlib import Path

from tau2.domains.retail.environment import get_tasks

TASKS = Path(__file__).resolve().parent.parent / "tasks"
READ = {"calculate", "find_user_id_by_email", "find_user_id_by_name_zip", "get_item_details",
        "get_order_details", "get_product_details", "get_user_details", "list_all_product_types",
        "think", "transfer_to_human_agents"}


def sig(t: dict) -> set[tuple]:
    return {(a["name"], a["arguments"].get("order_id", a["arguments"].get("user_id")))
            for a in t["evaluation_criteria"]["actions"] if a["name"] not in READ}


def main() -> int:
    hard = json.loads((TASKS / "tb500_hard.json").read_text())
    split = json.loads((TASKS / "split_hard.json").read_text())
    vary = json.loads((TASKS / "tb_vary.json").read_text())
    gen = json.loads((TASKS / "tb_gen.json").read_text())
    held = set().union(*(sig(x.model_dump()) for x in get_tasks()))

    # includes 3 hardened train tasks (tb_0032, tb_0219, tb_0237) whose default-address change
    # hits a user a held-out task re-addresses; p4-b and p4-c trained on them
    leaks = {t["id"] for t in hard + vary + gen if sig(t) & held and t["id"] not in set(split["val"])}
    hard = [t for t in hard if t["id"] not in leaks]
    split = {"train": [i for i in split["train"] if i not in leaks], "val": split["val"]}
    gen_keep = [t for t in gen if t["id"] not in leaks]
    vary_keep = [t for t in vary if t["id"] not in leaks]
    assert not any(t["id"].split("_r")[0] in set(split["val"]) for t in vary_keep), "variant of a val task"

    tasks = hard + vary_keep + gen_keep
    ids = [t["id"] for t in tasks]
    assert len(ids) == len(set(ids)), "duplicate task ids"
    new = {"train": sorted(split["train"] + [t["id"] for t in vary_keep + gen_keep]),
           "val": sorted(split["val"])}
    new["all"] = sorted(new["train"] + new["val"])
    assert not set(new["train"]) & set(new["val"])
    (TASKS / "tb_train_v4.json").write_text(json.dumps(tasks, indent=1))
    (TASKS / "split_v4.json").write_text(json.dumps(new, indent=1))
    print(f"v4: {len(tasks)} tasks | train {len(new['train'])} = hardened {len(split['train'])} "
          f"+ variants {len(vary_keep)} + generated {len(gen_keep)} | val {len(new['val'])} | "
          f"dropped for overlap with the 114: {sorted(leaks)[:10]}{'...' if len(leaks) > 10 else ''} "
          f"({len(leaks)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
