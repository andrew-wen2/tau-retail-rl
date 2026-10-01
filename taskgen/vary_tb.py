"""Re-rolled variants of the hardened TRAIN tasks, for variety (Sept 26).

p4-b and p4-c overfit the ~250-400 hardened train tasks: train reward kept rising while val
fell. Every transform in harden_tb.py decides whether and how
to apply by hashing the task id, so running the same chain under a new id gives the same
base request with a different mix of devices: different vague targets, decoys, changes of
mind, info questions and address requests. K variants per train task multiply the distinct
device combinations the policy sees, without touching val or the held-out 114 (variants
share their base task's orders, which were already deduped against the 114).

Usage (repo root):
    VARY_K=2 python3 taskgen/vary_tb.py    # K variants per train task, default 2
writes tb_vary.json (variants only, ids "<base>_r<k>") and vary_report.json. Gold replay is
validated on the VM like tb500_hard.json.

    VARY_SPLIT=val VARY_K=4 python3 taskgen/vary_tb.py
does the same for the 60 VAL tasks and writes tb_vary_val.json / vary_val_report.json: a
bigger val set (Sept 26) for when 60 tasks x 8 cannot resolve a checkpoint's gain. Val
variants share only their own val base's orders, so they never touch training.
"""

from __future__ import annotations

import json
import os

import harden_tb as H  # reads sys.argv[1] as a DB path at import, so K comes from the env

K = int(os.environ.get("VARY_K", "2"))
SPLIT = os.environ.get("VARY_SPLIT", "train")
SUFFIX = "" if SPLIT == "train" else f"_{SPLIT}"
FIELDS = {"vague_targets": 0, "change_of_mind": False, "stated_amount": None,
          "forbidden_first": None, "decoy": False, "conflict": None, "add_at_confirm": None,
          "info_question": None, "by_destination": False, "default_address": None,
          "twin_items": False}


def harden(t0: dict, st: H.Store, rep: dict) -> dict:
    """harden_tb.main's single-task chain, in the same order."""
    t = json.loads(json.dumps(t0))
    H.apply_decoy(t, st, rep)
    H.apply_by_destination(t, st, rep)
    H.apply_vague(t, st, rep)
    H.apply_twin_items(t, st, rep)
    H.apply_conflict(t, st, rep)
    H.apply_add_at_confirm(t, st, rep)
    H.apply_mind(t, st, rep)
    H.apply_forbidden(t, st, rep)
    H.apply_amount(t, st, rep)
    H.apply_info_question(t, st, rep)
    H.apply_default_address(t, st, rep)
    return t


def main() -> int:
    split = json.loads((H.TASKS / "split_tb500.json").read_text())
    want = set(split[SPLIT])
    base = [t for t in json.loads((H.TASKS / "tb500_descripted.json").read_text()) if t["id"] in want]
    out, report = [], {}
    # the existing hardened version of each task counts as already seen
    seen = {json.dumps([t["user_scenario"], t["evaluation_criteria"]], sort_keys=True)
            for t in json.loads((H.TASKS / "tb500_hard.json").read_text())}
    st = H.Store(json.loads(H.DB_PATH.read_text()))
    for t0 in base:
        for k in range(1, K + 1):
            t = json.loads(json.dumps(t0))
            t["id"] = f"{t0['id']}_r{k}"
            rep = dict(FIELDS, variant_of=t0["id"])
            t = harden(t, st, rep)
            # a variant identical to one already kept (no transform fired differently) adds nothing
            sig = json.dumps([t["user_scenario"], t["evaluation_criteria"]], sort_keys=True)
            if sig in seen:
                continue
            seen.add(sig)
            out.append(t)
            report[t["id"]] = rep
    (H.TASKS / f"tb_vary{SUFFIX}.json").write_text(json.dumps(out, indent=1))
    (H.TASKS / f"vary{SUFFIX}_report.json").write_text(json.dumps(report, indent=1))
    devs = {f: sum(bool(r[f]) for r in report.values()) for f in FIELDS}
    print(f"{len(out)} variants of {len(base)} {SPLIT} tasks (K={K}) | devices {devs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
