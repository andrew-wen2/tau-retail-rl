"""v11 (Sept 29, run p4-k): v10's pool with a sampler prior refreshed cheaply. Run on the VM in
.venv-tau2.

p4-j wasted 96 of 288 groups (33%) on tasks its policy solved every time: the prior for ~660 of
the tau3-style tasks rested on 2 attempts by older checkpoints, and 221 tasks had none. A full
re-profile (1,724 tasks x 8) is expensive; this is the light version:
  1. p4-j's own rollouts (policies one to four steps from soup4, the run's start, already under
     the tau3 reward) are added to the prior -- free;
  2. only tasks whose estimate is both thin and optimistic get profiled: fewer than 4 attempts
     with no failure seen, or never attempted. Those are the ones the sampler over-draws as
     "probably mixed" and that come back 8/8.
Writes tb_train_v11.json (= v10), split_v11.json (= v10), profile-v11-prior.json and
to_profile_v11.json for profile_v6.py (PROFILE_TAG=v11), which then writes profile-v11.json.
"""

from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path

import sys; sys.path.insert(0, ".")  # run from the repo root as taskgen/<script>
from records import iter_episodes


def main() -> int:
    shutil.copy("tasks/tb_train_v10.json", "tasks/tb_train_v11.json")
    shutil.copy("tasks/split_v10.json", "tasks/split_v11.json")
    train = json.load(open("tasks/split_v10.json"))["train"]
    prior = json.load(open("tasks/profile-v10.json"))
    added = Counter()
    for d in sorted(Path("runs").glob("p4-j*/records/b*")):
        for e in iter_episodes(d):
            if e.needs_reroll or not e.trains:
                continue
            p = prior.setdefault(e.task_id, {"successes": 0, "trials": 0})
            p["successes"] += e.eval_reward
            p["trials"] += 1
            added[e.task_id] += 1
    json.dump(prior, open("tasks/profile-v11-prior.json", "w"), indent=1)
    never = [t for t in train if t not in prior]
    thin = [t for t in train if t in prior and prior[t]["trials"] < 4
            and prior[t]["successes"] == prior[t]["trials"]]
    todo = sorted(never + thin)
    json.dump(todo, open("tasks/to_profile_v11.json", "w"))
    fam = Counter(t.split("_")[0] + ("_" + t.split("_")[1] if t.startswith("t6") else "") for t in todo)
    print(f"v11: prior +{sum(added.values())} p4-j episodes on {len(added)} tasks | to profile "
          f"{len(todo)} of {len(train)} (never tried {len(never)}, thin and all-success {len(thin)}) "
          f"{dict(fam)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
