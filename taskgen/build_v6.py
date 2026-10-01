"""Build the v6 training set (Sept 27, run p4-f). Run on the VM in .venv-tau2 (needs tau2).

  1. rebalance v5's train tasks toward tau3's mix (and drop tau2-gen's 150): changes of mind 34% -> ~12% (tau3 11%) and
     cancel-only tasks -> ~16% (tau3 16%); drop gen_targeted's level-1 tasks of the two families
     p4-e promoted past level 1 on its first step (items_addr, keep_options: >= 85% solved)
  2. add gen_v6.py's tasks after schema check, gold replay (writes must change the DB; no-write
     tasks must carry communicate_info and leave it unchanged), no write signature shared with the
     held-out 114 or any val task, and order-invariance of multi-item swaps (tau2 bug, build_v5)
  3. hold out ~15% of the v6 tasks per family as val_hard: tasks none of whose write
     signatures any training task shares
  4. write profile-v6-prior.json: profile-hard's counts, plus each vary_tb variant inheriting its
     base task's counts; and to_profile.json, the new tasks the starting adapter must be run on
     before training (the p4-e lesson: unprofiled tasks are drawn at a prior of p~0.7 and most
     are easy)

Writes tb_train_v6.json, split_v6.json (train, val = valbig + val_hard, val_hard),
profile-v6-prior.json, to_profile.json.
"""

from __future__ import annotations

import json
import random
from collections import Counter

from tau2.data_model.tasks import Task
from tau2.domains.retail.environment import get_environment, get_tasks

from build_v5 import READ, replay, reversed_lists, sig

CANCEL = "cancel_pending_order"


def writes(t: dict) -> list[dict]:
    return [a for a in t["evaluation_criteria"]["actions"] if a["name"] not in READ]


def users_of(t: dict, orders: dict) -> set[str]:  # noqa: unused since val_hard went by signature
    out = set()
    for a in writes(t):
        if "user_id" in a["arguments"]:
            out.add(a["arguments"]["user_id"])
        if a["arguments"].get("order_id") in orders:
            out.add(orders[a["arguments"]["order_id"]]["user_id"])
    return out


def main() -> int:
    rng = random.Random(606)
    orders = json.load(open("data/retail/db.json"))["orders"]      # tau2 b7ea907's, pinned
    v5 = {t["id"]: t for t in json.load(open("tasks/tb_train_v5.json"))}
    sp5 = json.load(open("tasks/split_v5.json"))
    rep = {**json.load(open("tasks/harden_report.json")), **json.load(open("tasks/vary_report.json"))}
    valbig = [v5[t] for t in sp5["val"]]
    held = set().union(*(sig(x.model_dump()) for x in get_tasks()))
    val_sig = set().union(*(sig(t) for t in valbig))
    base_hash = get_environment().get_db_hash()

    # 1. rebalance
    # tau2-gen's tasks are dropped too: mostly single-action, 4 of 9 all-success at p4-e step 0,
    # and unprofiled -- not worth the profiling spend
    train = [v5[t] for t in sp5["train"]
             if not t.startswith(("tt_items_addr_L1_", "tt_keep_options_L1_", "tg_"))]
    com = [t for t in train if (rep.get(t["id"]) or {}).get("change_of_mind")]
    cancel = [t for t in train if writes(t) and all(a["name"] == CANCEL for a in writes(t))
              and t not in com]

    # 2. v6 tasks
    kept, drop = [], Counter()
    for t in json.load(open("tasks/tb_v6.json")):
        try:
            Task.model_validate(t)
            h = replay(t["evaluation_criteria"]["actions"])
        except Exception as e:  # noqa: BLE001
            drop[f"schema/replay: {str(e)[:40]}"] += 1
            continue
        if writes(t) and h == base_hash:
            drop["writes but no DB change"] += 1
            continue
        if not writes(t) and (h != base_hash or not t["evaluation_criteria"]["communicate_info"]):
            drop["no-write task without a stated fact"] += 1
            continue
        if sig(t) & held:
            drop["shares a write with the 114"] += 1
            continue
        if sig(t) & val_sig:
            drop["shares a write with val"] += 1
            continue
        acts = t["evaluation_criteria"]["actions"]
        if any("new_item_ids" in a["arguments"] and len(a["arguments"]["item_ids"]) > 1 for a in acts):
            if replay(reversed_lists(acts)) != h:
                drop["end state depends on item order"] += 1
                continue
        kept.append(t)

    # 3. val_hard: v6 tasks none of whose writes (tool, order/user) any other training task makes
    base_sig = set().union(*(sig(t) for t in train))
    n_sig = Counter(x for t in kept for x in sig(t))
    cand = [t for t in kept if not sig(t) & base_sig and all(n_sig[x] == 1 for x in sig(t))]
    rng.shuffle(cand)
    want = {f: max(4, round(0.15 * sum(t["id"].startswith(f"t6_{f}_") for t in kept)))
            for f in ("bundle", "single", "choose", "refuse")}
    val_hard, got = [], Counter()
    for t in cand:
        f = t["id"].split("_")[1]
        if got[f] < want[f]:
            val_hard.append(t)
            got[f] += 1
    vh = {t["id"] for t in val_hard}
    new_train = [t for t in kept if t["id"] not in vh]

    # rebalance against the FINAL train size
    pool = train + new_train
    n = len(pool)
    drop_com = rng.sample(com, max(0, len(com) - round(0.12 * n)))
    drop_cancel = rng.sample(cancel, max(0, len(cancel) - round(0.16 * n)))
    gone = {t["id"] for t in drop_com + drop_cancel}
    pool = [t for t in pool if t["id"] not in gone]

    tasks = pool + valbig + val_hard
    ids = [t["id"] for t in tasks]
    assert len(ids) == len(set(ids)), "duplicate ids"
    split = {"train": sorted(t["id"] for t in pool),
             "val": sorted(t["id"] for t in valbig + val_hard),
             "val_hard": sorted(vh)}
    assert not set(split["train"]) & set(split["val"])
    json.dump(tasks, open("tasks/tb_train_v6.json", "w"), indent=1)
    json.dump(split, open("tasks/split_v6.json", "w"), indent=1)

    # 4. priors and what still needs profiling
    prof = json.load(open("tasks/profile-hard.json"))
    prior = dict(prof)
    for t in pool:
        base = t["id"].split("_r")[0]
        if "_r" in t["id"] and base in prof:
            prior[t["id"]] = dict(prof[base])
    json.dump(prior, open("tasks/profile-v6-prior.json", "w"), indent=1)
    to_profile = sorted(t["id"] for t in pool if t["id"] not in prior)
    json.dump(to_profile, open("tasks/to_profile.json", "w"))

    fam = Counter(t["id"].split("_")[1] if t["id"].startswith("t6_") else
                  ("targeted" if t["id"].startswith("tt_") else "generated" if t["id"].startswith("tg_")
                   else "variant" if "_r" in t["id"] else "hardened") for t in pool)
    w = [len(writes(t)) for t in pool]
    print(f"v6 train {len(pool)} {dict(fam)} | val {len(split['val'])} = valbig {len(valbig)} + "
          f"val_hard {len(vh)} {dict(got)}")
    print(f"dropped v6: {dict(drop)} | rebalance dropped change-of-mind {len(drop_com)}, "
          f"cancel-only {len(drop_cancel)}")
    print(f"train shares: 3+ writes {sum(x >= 3 for x in w) / len(w):.0%}, no-write "
          f"{sum(x == 0 for x in w) / len(w):.0%}, change of mind "
          f"{sum(bool((rep.get(t['id']) or {}).get('change_of_mind')) for t in pool) / len(pool):.0%}")
    print(f"to profile: {len(to_profile)} tasks (prior covers {sum(t['id'] in prior for t in pool)} of {len(pool)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
