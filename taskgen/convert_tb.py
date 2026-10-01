"""Convert the tau-bench retail TASKS_TRAIN into tau2 (tau-3) task format, replay-validate
each one's gold actions against the tau2 retail environment, and record what the replay
shows about the task's training value.

Two facts drive the design:

  * tau2's `UserInstructions` is `StructuredUserInstructions | str`, so tau-bench's single
    flat instruction string transfers verbatim. No LLM decomposition, no paraphrase risk.
  * tau2's DB evaluator replays the gold actions and *swallows* any exception with a
    warning (evaluator_env.py). A task whose gold actions fail therefore does not error --
    it silently becomes a task the agent wins by doing nothing. So every conversion is
    replayed here, with exceptions caught explicitly, and tasks are classified by whether
    the gold replay actually moves the database.

Emits tb500_retail.json (the converted set) and convert_report.json.
Runs on CPU in seconds; no GPU and no API key.
"""

import hashlib
import json
import re
import sys
from pathlib import Path

if len(sys.argv) < 2:
    sys.exit("usage: convert_tb.py DIR  (DIR contains a clone of sierra-research/tau-bench as tau-bench/)")
SCRATCH = Path(sys.argv[1])
sys.path.insert(0, str(SCRATCH / "tau-bench"))

from tau_bench.envs.retail.tasks_train import TASKS_TRAIN  # noqa: E402

from tau2.data_model.tasks import Task  # noqa: E402
from tau2.domains.retail.environment import get_environment, get_tasks  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "tasks"

# Retail tools that only read. Anything else can mutate the DB. Used only to explain a
# result; the authoritative signal is the measured db-hash change during replay.
READ_ONLY = {
    "calculate", "find_user_id_by_email", "find_user_id_by_name_zip",
    "get_order_details", "get_product_details", "get_user_details",
    "list_all_product_types", "think", "transfer_to_human_agents",
}


def convert(idx: int, tb_task) -> dict:
    """tau-bench Task -> tau2 Task dict. reward_basis [DB, COMMUNICATE] is tau2's own
    default and is documented there as 'matching the original tau-bench', so the converted
    tasks keep tau-bench's reward semantics exactly. Note this means NO nl_assertions and
    therefore NO LLM judge on any converted task."""
    tid = f"tb_{idx:04d}"
    return {
        "id": tid,
        "description": {
            "purpose": None,
            "relevant_policies": None,
            "notes": f"Converted from tau-bench retail TASKS_TRAIN[{idx}] "
                     f"(user_id={tb_task.user_id}).",
        },
        "user_scenario": {"persona": None, "instructions": tb_task.instruction},
        "initial_state": None,
        "evaluation_criteria": {
            "actions": [
                {
                    "action_id": f"{tid}_{j}",
                    "requestor": "assistant",
                    "name": a.name,
                    "arguments": a.kwargs,
                    "info": None,
                    "compare_args": None,
                }
                for j, a in enumerate(tb_task.actions)
            ],
            "communicate_info": list(tb_task.outputs),
            "nl_assertions": None,
            "reward_basis": ["DB", "COMMUNICATE"],
        },
    }


def replay(task_dict: dict) -> dict:
    """Replay the gold actions in a fresh env. Returns what the replay proves."""
    env = get_environment()
    before = env.get_db_hash()
    errors = []
    for action in task_dict["evaluation_criteria"]["actions"]:
        try:
            env.make_tool_call(
                tool_name=action["name"],
                requestor=action["requestor"],
                **action["arguments"],
            )
        except Exception as e:  # noqa: BLE001 - we want every failure mode recorded
            errors.append({"action": action["name"], "error": f"{type(e).__name__}: {e}"})
    after = env.get_db_hash()
    return {"db_before": before, "db_after": after, "db_mutated": before != after,
            "replay_errors": errors}


# ---- near-duplicate / leakage check against all 114 held-out tau2 tasks ----

ID_RE = re.compile(r"#W\d+|\b\d{10}\b|[a-z]+_[a-z]+_\d+")


def entities(text: str) -> frozenset:
    """Order ids (#W...), item/product ids (10 digits), user ids (name_name_dddd)."""
    return frozenset(ID_RE.findall(text or ""))


def write_signature(actions) -> frozenset:
    """(write tool, order id) pairs. Two tasks sharing one share an answer, however
    differently they are worded -- this is the check that actually catches leakage."""
    sig = set()
    for a in actions:
        name = a["name"] if isinstance(a, dict) else a.name
        args = a["arguments"] if isinstance(a, dict) else a.arguments
        if name in READ_ONLY:
            continue
        key = args.get("order_id") or args.get("user_id") or ""
        sig.add((name, key))
    return frozenset(sig)


def held_out_text(t: Task) -> str:
    ins = t.user_scenario.instructions
    return ins if isinstance(ins, str) else str(ins)


def main() -> None:
    held_out = get_tasks("base")
    assert len(held_out) == 114, f"expected 114 held-out tasks, got {len(held_out)}"
    ho = [
        {
            "id": t.id,
            "entities": entities(held_out_text(t)),
            "wsig": write_signature(t.evaluation_criteria.actions or []),
        }
        for t in held_out
    ]

    converted, report = [], []
    for i, tb in enumerate(TASKS_TRAIN):
        d = convert(i, tb)
        try:
            Task.model_validate(d)          # schema must be exactly right
            schema_ok, schema_err = True, None
        except Exception as e:              # noqa: BLE001
            schema_ok, schema_err = False, f"{type(e).__name__}: {e}"

        r = replay(d) if schema_ok else {"db_mutated": False, "replay_errors": [],
                                         "db_before": None, "db_after": None}

        ents = entities(tb.instruction)
        wsig = write_signature(d["evaluation_criteria"]["actions"])
        # leakage: a held-out task that performs the same write on the same object
        collisions = [h["id"] for h in ho if wsig and (wsig & h["wsig"])]
        # softer signal: heavy shared-entity overlap with a held-out task
        overlap = max(
            (len(ents & h["entities"]) / max(1, len(ents | h["entities"])) for h in ho),
            default=0.0,
        )

        n_write = sum(
            1 for a in d["evaluation_criteria"]["actions"] if a["name"] not in READ_ONLY
        )
        n_comm = len(d["evaluation_criteria"]["communicate_info"])

        rec = {
            "id": d["id"], "schema_ok": schema_ok, "schema_err": schema_err,
            "replay_errors": r["replay_errors"], "db_mutated": r["db_mutated"],
            "n_actions": len(d["evaluation_criteria"]["actions"]),
            "n_write_actions": n_write, "n_communicate": n_comm,
            "instruction_chars": len(tb.instruction),
            "write_collisions": collisions, "max_entity_overlap": round(overlap, 3),
        }
        # A task carries training signal only if the agent can get it wrong: either the
        # gold replay moves the DB, or something must be told to the customer. Otherwise
        # reward is 1.0 for any behaviour at all, including doing nothing.
        rec["has_signal"] = bool(r["db_mutated"] or n_comm)
        rec["keep"] = bool(
            schema_ok and not r["replay_errors"] and rec["has_signal"] and not collisions
        )
        report.append(rec)
        if rec["keep"]:
            converted.append(d)

    (OUT / "tb500_retail.json").write_text(json.dumps(converted, indent=1))
    (OUT / "convert_report.json").write_text(json.dumps(report, indent=1))

    n = len(report)
    def c(pred): return sum(1 for r in report if pred(r))
    print(f"tau-bench retail TASKS_TRAIN: {n}")
    print(f"  schema invalid          {c(lambda r: not r['schema_ok'])}")
    print(f"  gold replay errored     {c(lambda r: r['replay_errors'])}")
    print(f"  no signal (free reward) {c(lambda r: r['schema_ok'] and not r['has_signal'])}")
    print(f"    of which read-only    {c(lambda r: not r['db_mutated'] and not r['n_communicate'])}")
    print(f"  write-collides with 114 {c(lambda r: r['write_collisions'])}")
    print(f"  KEPT                    {len(converted)}")
    print(f"\nwrote {OUT/'tb500_retail.json'} and {OUT/'convert_report.json'}")


if __name__ == "__main__":
    main()
