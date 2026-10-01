"""Profile the tasks the v6 sampler has no prior for (Sept 27): serve the run's starting adapter,
run each task in to_profile.json G times with the DeepSeek customer, and write
profile-v6.json = profile-v6-prior.json + these counts. Run on the VM in .venv-train with the
training server up (baseline/session.sh), API_BUDGET_USD set.

Why the starting adapter and not the base model: the sampler's job is to find tasks the policy
it is about to train will get mixed results on; p4-e drew unprofiled variants at a prior of
p~0.7 and 18 of 25 came back 8/8 for this adapter.

Usage: python taskgen/profile_v6.py <adapter dir> [G] [id prefix]   (prefix: profile only matching ids)
Env PROFILE_MERGE=add adds the new counts to the prior's instead of replacing them.
Env PROFILE_TAG=v7 reads to_profile_v7.json / profile-v7-prior.json / tb_train_v7.json and writes
profile-v7.json (run p4-g).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import sys; sys.path.insert(0, ".")  # run from the repo root as taskgen/<script>
from records import iter_episodes
from trainer import HERE, TAU2_PYTHON, Config, swap_adapter


def main() -> int:
    adir = Path(sys.argv[1])
    G = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    import os
    tag = os.environ.get("PROFILE_TAG", "v6")
    suffix = "" if tag == "v6" else "_" + tag
    tasks = json.load(open(f"tasks/to_profile{suffix}.json"))
    if len(sys.argv) > 3:
        tasks = [t for t in tasks if t.startswith(sys.argv[3])]
    cfg = Config()
    serving = swap_adapter(cfg, adir, "prof-" + adir.name)
    out = HERE / "runs" / f"profile-{tag}"
    m = {"batch_id": 300000, "policy_version": adir.name, "model": serving, "tasks": tasks, "G": G,
         "seed": 606, "out_dir": str(out), "concurrency": 128,
         "customer_model": cfg.customer_model, "base_url": cfg.base_url,
         "guards": {"max_steps": 80, "max_tokens_per_turn": 4096}, "tasks_file": f"tb_train_{tag}.json"}
    mp = HERE / "runs" / f"profile-{tag}.json.manifest"
    mp.write_text(json.dumps(m, indent=1))
    proc = subprocess.run([str(TAU2_PYTHON), str(HERE / "driver.py"), "batch", str(mp)],
                          capture_output=True, text=True, cwd=HERE)
    if proc.returncode != 0:
        print(proc.stderr[-3000:], file=sys.stderr)
        return 1
    counts: dict[str, dict[str, int]] = {}
    for e in iter_episodes(out):
        if e.needs_reroll:
            continue
        c = counts.setdefault(e.task_id, {"successes": 0, "trials": 0})
        c["successes"] += e.eval_reward
        c["trials"] += 1
    prior = json.load(open(f"tasks/profile-{tag}-prior.json"))
    if os.environ.get("PROFILE_MERGE") == "add":
        # Sept 29 (v11): a light re-profile adds to the thin existing counts instead of
        # replacing them, so a task's earlier attempts still count
        for t, c in counts.items():
            p = prior.setdefault(t, {"successes": 0, "trials": 0})
            p["successes"] += c["successes"]
            p["trials"] += c["trials"]
    else:
        prior.update(counts)
    json.dump(prior, open(f"tasks/profile-{tag}.json", "w"), indent=1)
    by = {}
    for t, c in counts.items():
        fam = t.split("_")[1] if t.startswith("t6_") else t.split("_")[0]
        by.setdefault(fam, []).append(c["successes"] / c["trials"])
    print(proc.stdout.strip().splitlines()[-1][:300])
    for f, ps in sorted(by.items()):
        print(f"  {f:<10} n={len(ps):>3}  mean {sum(ps)/len(ps):.2f}  always {sum(p == 1 for p in ps)/len(ps):.0%}"
              f"  never {sum(p == 0 for p in ps)/len(ps):.0%}  mixed {sum(0 < p < 1 for p in ps)/len(ps):.0%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
