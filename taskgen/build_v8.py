"""v8 (Sept 28, run p4-i): v7 plus a fresh batch aimed at p4-h's weakest targets. Run on the VM in .venv-tau2.

Same checks as v7 (below). The batch (tb_v6c.json, ids t6c_*) is gen_v6.py with V6_STYLES=relative,partial
and V6_SUITE=1: in p4-h's rollouts "keep the other options" targets succeeded 0.58 and relative
targets 0.60 against 0.82 for exact and cheapest/most-expensive, and looked-up addresses were
copied with the old suite (address2 55 of the wrong arguments). Tasks whose gold writes duplicate
a v6/v6b task are dropped. The prior adds p4-h's step-0 batch, whose policy was p4-f step 8.

---- v7 notes ----

p4-f's pool was thinning by step 6 (16 of 32 groups degenerate) and bundles -- 3-5 writes across
2-3 orders, the family where the adapter is weakest (0.60 profiled) -- were the scarcest. This
adds gen_v6.py's second batch of bundles (tb_v6b.json, ids t6b_bundle_*) after the same checks as
build_v6 (schema, gold replay, DB change, no write shared with the held-out 114 or any val task,
order-invariant swaps), and writes the sampler prior as profile-v6.json plus every attempt p4-f
made in training, so tasks p4-f already learned to solve are seen as solved.

Val is the 166-task set only (split "val"), read at 8 attempts.
Writes tb_train_v7.json, split_v7.json, profile-v7-prior.json, to_profile_v7.json.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from tau2.data_model.tasks import Task
from tau2.domains.retail.environment import get_environment, get_tasks

from build_v5 import replay, reversed_lists, sig
from build_v6 import writes
import sys; sys.path.insert(0, ".")  # run from the repo root as taskgen/<script>
from records import iter_episodes


def main() -> int:
    v6 = {t["id"]: t for t in json.load(open("tasks/tb_train_v7.json"))}
    sp6f = json.load(open("tasks/split_v7.json"))
    sp6 = json.load(open("tasks/split_v6.json"))
    valbig = sp6f["val"]
    held = set().union(*(sig(x.model_dump()) for x in get_tasks()))
    val_sig = set().union(*(sig(v6[t]) for t in valbig)) | set().union(*(sig(t) for t in json.load(open("tasks/tb_train_v6.json")) if t["id"] in set(sp6["val_hard"])))
    base_hash = get_environment().get_db_hash()
    canon = lambda t: json.dumps(sorted(json.dumps([a["name"], a["arguments"]], sort_keys=True)  # noqa: E731
                                        for a in t["evaluation_criteria"]["actions"]))
    seen_gold = {canon(t) for t in v6.values()}
    new, drop = [], Counter()
    for t in json.load(open("tasks/tb_v6c.json")):
        try:
            Task.model_validate(t)
            h = replay(t["evaluation_criteria"]["actions"])
        except Exception as e:  # noqa: BLE001
            drop[f"schema/replay: {str(e)[:40]}"] += 1
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
            if replay(reversed_lists(acts)) != h:
                drop["end state depends on item order"] += 1
                continue
        if canon(t) in seen_gold:
            drop["same gold writes as an existing task"] += 1
            continue
        new.append(t)
    train = sp6f["train"] + [t["id"] for t in new]
    tasks = [v6[t] for t in sp6f["train"]] + new + [v6[t] for t in valbig]
    ids = [t["id"] for t in tasks]
    assert len(ids) == len(set(ids)), "duplicate ids"
    json.dump(tasks, open("tasks/tb_train_v9.json", "w"), indent=1)
    json.dump({"train": sorted(train), "val": sorted(valbig), "all": sorted(train + valbig)},
              open("tasks/split_v9.json", "w"), indent=1)
    prior = json.load(open("tasks/profile-v8.json"))
    seen = Counter()
    for d in [Path("runs/p4-h/records/b0000")]:
        for e in iter_episodes(d):
            if e.needs_reroll or e.task_id not in prior:
                continue
            prior[e.task_id]["successes"] += e.eval_reward
            prior[e.task_id]["trials"] += 1
            seen[e.task_id] += 1
    json.dump(prior, open("tasks/profile-v9-prior.json", "w"), indent=1)
    json.dump(sorted(t["id"] for t in new), open("tasks/to_profile_v9.json", "w"))
    w = Counter(len(writes(t)) for t in new)
    print(f"v8 train {len(train)} = v7 pool {len(sp6f['train'])} + new bundles {len(new)} "
          f"(writes {dict(sorted(w.items()))}) | val {len(valbig)}")
    print(f"dropped: {dict(drop)} | p4-f attempts folded into the prior: {sum(seen.values())} on "
          f"{len(seen)} tasks (p4-h step 0 = p4-f step 8)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
