"""v12 (Sept 29, run p4-l): v11's tasks, with the pool cut to tasks the CURRENT policy gets mixed.
Run on the VM in .venv-tau2.

Research round 2: RAFT and AReaL-SEA filter their training
tasks by difficulty under the policy being trained, and our own only big jump (p4-f) came from
profiling every task with the current adapter first. The v11 prior pools every attempt since
the base model, so a task base failed 8/8 and soup4 now solves 8/8 still reads "mixed", and
p4-k spent 22% of its groups on degenerate tasks.

The fix uses rollouts already paid for: every episode from soup4's lineage -- the v11 light
profile (soup4 itself), p4-j1..3 and p4-k (one to six steps from soup4) -- is the "recent"
profile. A task with >= RECENT_MIN recent attempts is kept only if they were mixed; a task with
fewer falls back to the full prior and is kept if that was mixed. The sampler prior becomes the
recent counts where they are thick enough, else the full ones.

Writes tb_train_v12.json (= v11), split_v12.json (train = the filtered pool) and profile-v12.json.
"""

from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path

import sys; sys.path.insert(0, ".")  # run from the repo root as taskgen/<script>
from records import iter_episodes

RECENT_MIN = 4
RECENT_DIRS = ["runs/profile-v11"] + [f"runs/{r}/records" for r in ("p4-j1", "p4-j2", "p4-j3", "p4-k")]


def main() -> int:
    shutil.copy("tasks/tb_train_v11.json", "tasks/tb_train_v12.json")
    sp = json.load(open("tasks/split_v11.json"))
    train = sp["train"]
    full = json.load(open("tasks/profile-v11.json"))
    # p4-k's rollouts are not in profile-v11 yet (p4-j's and the light profile's are)
    for e in (e for d in sorted(Path("runs/p4-k/records").glob("b*")) for e in iter_episodes(d)):
        if e.needs_reroll or not e.trains:
            continue
        p = full.setdefault(e.task_id, {"successes": 0, "trials": 0})
        p["successes"] += e.eval_reward
        p["trials"] += 1

    recent: dict[str, dict] = {}
    dirs = [Path(d) for d in RECENT_DIRS if Path(d).is_dir()]
    batch_dirs = [b for d in dirs for b in ([x for x in sorted(d.glob("b*")) if x.is_dir()] or [d])]
    for d in batch_dirs:
        for e in iter_episodes(d):
            if e.needs_reroll or not e.trains:
                continue
            p = recent.setdefault(e.task_id, {"successes": 0, "trials": 0})
            p["successes"] += e.eval_reward
            p["trials"] += 1

    mixed = lambda p: p is not None and 0 < p["successes"] < p["trials"]  # noqa: E731
    keep, why = [], Counter()
    prior: dict[str, dict] = {}
    for t in train:
        r, f = recent.get(t), full.get(t)
        if r and r["trials"] >= RECENT_MIN:
            prior[t] = r
            if mixed(r):
                keep.append(t); why["recent mixed"] += 1
            else:
                why["recent always" if r["successes"] else "recent never"] += 1
        else:
            if f:
                prior[t] = f
            if mixed(f):
                keep.append(t); why["full mixed (recent thin)"] += 1
            else:
                why["full not mixed (recent thin)"] += 1
    json.dump({"train": sorted(keep), "val": sp["val"], "all": sorted(keep + sp["val"])},
              open("tasks/split_v12.json", "w"), indent=1)
    json.dump(prior, open("tasks/profile-v12.json", "w"), indent=1)
    rate = sum(prior[t]["successes"] for t in keep) / max(1, sum(prior[t]["trials"] for t in keep))
    print(f"v12: {len(keep)} of {len(train)} train tasks kept | recent episodes "
          f"{sum(p['trials'] for p in recent.values())} on {len(recent)} tasks from "
          f"{len(batch_dirs)} dirs | {dict(why)} | kept-pool success {rate:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
