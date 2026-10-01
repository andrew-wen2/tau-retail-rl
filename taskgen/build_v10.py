"""v10 (Sept 28, runs p4-j1..3): the v9 training pool under tau3's reward rule. Run on the VM in
.venv-tau2.

Until Sept 28 the driver scored the DB alone; tau2 also requires every communicate_info string to
appear in an agent message (349 of v9's 1,742 training tasks carry one, 60 of them with no write
at all -- those were free reward). This drops the 18 tasks neither the 2B nor DeepSeek-V4.1-Flash
ever solved (unsolvable_tasks.json), and rebuilds the sampler prior from every recorded episode
of the hardened and v6..v9 profiles and runs p4-f..p4-i, re-scored as DB x COMMUNICATE.

Writes tb_train_v10.json (= v9's tasks), split_v10.json, profile-v10.json.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import sys; sys.path.insert(0, ".")  # run from the repo root as taskgen/<script>
from driver import communicated
from records import AssistantTurn, iter_episodes


def main() -> int:
    tasks = {t["id"]: t for t in json.load(open("tasks/tb_train_v9.json"))}
    sp9 = json.load(open("tasks/split_v9.json"))
    drop = set(json.load(open("tasks/unsolvable_tasks.json")))
    train = sorted(t for t in sp9["train"] if t not in drop)
    json.dump(list(tasks.values()), open("tasks/tb_train_v10.json", "w"), indent=1)
    json.dump({"train": train, "val": sp9["val"], "all": sorted(train + sp9["val"])},
              open("tasks/split_v10.json", "w"), indent=1)

    pool = set(train)
    prof: dict[str, dict] = {}
    flipped = Counter()
    # profiling runs write their episodes flat in one directory; training runs, one per batch
    dirs = [d for d in ("runs/profile-hard", "runs/profile-v6", "runs/profile-v7", "runs/profile-v9")
            if Path(d).is_dir()]
    dirs += sorted(str(d) for pat in ("runs/p4-f/records", "runs/p4-g/records",
                                      "runs/p4-h/records", "runs/p4-i/records")
                   for d in Path(pat).glob("b*") if d.is_dir())
    for d in dirs:
        for e in iter_episodes(d):
            if e.needs_reroll or e.task_id not in pool:
                continue
            r = e.eval_reward
            if r:
                infos = tasks[e.task_id]["evaluation_criteria"].get("communicate_info") or []
                ok = communicated([t.text for t in e.turns if isinstance(t, AssistantTurn)], infos)
                flipped["success -> unsaid"] += not ok
                r = int(ok)
            p = prof.setdefault(e.task_id, {"successes": 0, "trials": 0})
            p["successes"] += r
            p["trials"] += 1
    json.dump(prof, open("tasks/profile-v10.json", "w"), indent=1)
    n = [p["trials"] for p in prof.values()]
    comm = [t for t in train if tasks[t]["evaluation_criteria"].get("communicate_info")]
    rate = lambda ts: (sum(prof[t]["successes"] for t in ts if t in prof)
                       / max(1, sum(prof[t]["trials"] for t in ts if t in prof)))
    print(f"v10 train {len(train)} (dropped {len(drop & set(sp9['train']))} unsolvable) | "
          f"profiled {len(prof)} tasks, {sum(n)} episodes from {len(dirs)} batches | "
          f"{dict(flipped)}")
    print(f"success under the tau3 rule: all {rate(train):.3f} | with communicate_info "
          f"{rate(comm):.3f} ({len(comm)} tasks) | without {rate([t for t in train if t not in set(comm)]):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
