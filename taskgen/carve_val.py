"""Carve a training-time validation split out of the converted tau-bench tasks.

Why this exists: without it, tuning decisions have nowhere to get their feedback, and
every "did that checkpoint help?" would have had to either burn a full run on the 114
held-out tasks or contaminate them. Val fixes that. The 114 are touched exactly twice --
once for the base arm (done) and once for the final trained arm.

Stratification. The ideal stratifier is each task's measured base success count, but that
needs a GPU rollout pass that has not run yet. Until it has, this stratifies on the two
structural properties that drive retail difficulty and are known today:

    primary write tool   cancel / exchange / return / modify_items / modify_address / ...
    n_write_actions      1 / 2 / 3+

Task type matters more than raw action count here: the held-out 114 and the converted pool
have the same median write count (1), but their mix of write tools differs, and the tool is
what determines the policy path the agent has to follow.

Once the profiling pass exists, re-run with --profile <profile.json> and it re-stratifies
on measured success counts instead; --check reports whether the structural split already
matches on that basis, so the reshuffle only happens if it is actually needed.

Deterministic: seed 300, the same seed the baselines used.
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

TASKS = Path(__file__).resolve().parent.parent / "tasks"
SEED = 300


READ_ONLY = {
    "calculate", "find_user_id_by_email", "find_user_id_by_name_zip",
    "get_order_details", "get_product_details", "get_user_details",
    "list_all_product_types", "think", "transfer_to_human_agents",
}


def stratum_structural(rec: dict) -> tuple:
    return (rec["primary_write_tool"], min(rec["n_write_actions"], 3))


def stratum_measured(rec: dict, profile: dict) -> tuple:
    """Bucket by measured base success count c/n, matching the buckets summarize.py uses:
    never-solved, low, band, always-solved."""
    p = profile[rec["id"]]
    c, n = p["successes"], p["trials"]
    frac = c / n
    if frac == 0:
        return ("never",)
    if frac == 1:
        return ("always",)
    if frac >= 0.6:
        return ("band",)
    return ("low",)


def carve(records, key, n_val):
    """Proportional allocation across strata, largest-remainder, deterministic."""
    rng = random.Random(SEED)
    strata = defaultdict(list)
    for r in records:
        strata[key(r)].append(r["id"])
    for ids in strata.values():
        ids.sort()
        rng.shuffle(ids)

    total = len(records)
    exact = {s: len(ids) * n_val / total for s, ids in strata.items()}
    alloc = {s: int(v) for s, v in exact.items()}
    # largest remainder, ties broken by stratum key so the result is reproducible
    for s in sorted(strata, key=lambda s: (-(exact[s] - alloc[s]), str(s))):
        if sum(alloc.values()) >= n_val:
            break
        if alloc[s] < len(strata[s]):
            alloc[s] += 1

    val = sorted(i for s, ids in strata.items() for i in ids[: alloc[s]])
    train = sorted(r["id"] for r in records if r["id"] not in set(val))
    return val, train, strata, alloc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-val", type=int, default=60)
    ap.add_argument("--profile", type=Path, default=None,
                    help="profile.json mapping task id -> {successes, trials}")
    ap.add_argument("--check", action="store_true",
                    help="with --profile: report drift of the existing split, do not rewrite")
    args = ap.parse_args()

    report = json.load(open(TASKS / "convert_report.json"))
    tasks = json.load(open(TASKS / "tb500_retail.json"))
    kept_ids = {t["id"] for t in tasks}
    records = [r for r in report if r["id"] in kept_ids]
    assert len(records) == len(tasks)

    # The primary write tool lives in the task file, not the report; attach it.
    by_task = {t["id"]: t for t in tasks}
    for r in records:
        writes = [a["name"] for a in by_task[r["id"]]["evaluation_criteria"]["actions"]
                  if a["name"] not in READ_ONLY]
        r["primary_write_tool"] = writes[0] if writes else "none"

    # Guardrails, asserted rather than assumed.
    for r in records:
        assert not r["write_collisions"], f"{r['id']} leaks into held-out {r['write_collisions']}"
        assert r["schema_ok"] and not r["replay_errors"] and r["has_signal"]

    if args.profile:
        profile = json.load(open(args.profile))
        key = lambda r: stratum_measured(r, profile)  # noqa: E731
        basis = "measured base success count"
    else:
        key = stratum_structural
        basis = "structural (primary write tool x n_write_actions)"

    if args.check:
        split = json.load(open(TASKS / "split_tb500.json"))
        val = set(split["val"])
        profile = json.load(open(args.profile))
        for name, ids in (("val", val), ("train", set(split["train"]))):
            dist = defaultdict(int)
            for r in records:
                if r["id"] in ids:
                    dist[stratum_measured(r, profile)[0]] += 1
            tot = sum(dist.values())
            print(name, {k: f"{v} ({100*v/tot:.0f}%)" for k, v in sorted(dist.items())})
        return

    val, train, strata, alloc = carve(records, key, args.n_val)

    split = {"val": val, "train": train, "all": sorted(kept_ids)}
    (TASKS / "split_tb500.json").write_text(json.dumps(split, indent=1))

    by_id = {r["id"]: r for r in records}
    print(f"converted & kept : {len(records)}")
    print(f"stratified on    : {basis}")
    print(f"val              : {len(val)}")
    print(f"train            : {len(train)}\n")
    print(f"{'stratum':<28}{'pool':>6}{'val':>6}{'train':>7}")
    for s in sorted(strata, key=str):
        nv = sum(1 for i in val if key(by_id[i]) == s)
        label = f"{s[0]} x{s[1]}" if len(s) == 2 else str(s[0])
        print(f"{label:<28}{len(strata[s]):>6}{nv:>6}{len(strata[s])-nv:>7}")
    print(f"\nwrote {TASKS/'split_tb500.json'}")

    # How long one val pass takes, so the cadence decision is grounded.
    eps = len(val) * 4
    print(f"\nval pass @ n=4 = {eps} episodes"
          f"  ~{eps*58/40/60:.0f} min GPU @ CONC=40")
    print("no LLM judge: converted tasks are reward_basis [DB, COMMUNICATE], both local")


if __name__ == "__main__":
    main()
