"""Train/eval fidelity check: run tau2's OWN orchestrator and
evaluator on the same tasks, against the same server, with the same customer, and compare
reward rate, turn count and termination mix with what driver.py recorded.

The driver re-implements tau2's turn loop so it can capture token IDs. If that loop drifted
from tau2's (a different opening, a tool result shaped differently, a limit counted
differently), the driver would train on episodes that eval never produces, and nothing else
in the loop would notice. This is the check that would.

Both sides go through the training server, but tau2's requests carry no reasoning_content
marker, so qwen35_train.jinja renders them exactly as the stock template does: this side is
the eval path. Customer spend is added to the driver's ledger.

Usage (.venv-tau2, repo root):
    python3 fidelity.py <driver_records_dir> --tasks tb_0001,tb_0002,... --trials 8
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from driver import HERE, _load_dotenv, load_tasks, record_spend, spent_usd, API_BUDGET_USD
from records import iter_episodes

AGENT_ARGS = {"api_base": "http://localhost:8000/v1", "presence_penalty": 1.5,
              "temperature": 0.7, "top_p": 0.8,
              "extra_body": {"top_k": 20, "chat_template_kwargs": {"enable_thinking": False}}}


def run_tau2(task_dict: dict, seed: int, max_steps: int, customer: str) -> dict:
    from tau2.agent.llm_agent import LLMAgent
    from tau2.data_model.tasks import Task
    from tau2.domains.retail.environment import get_environment
    from tau2.orchestrator.orchestrator import Orchestrator
    from tau2.runner.simulation import run_simulation
    from tau2.user.user_simulator import UserSimulator

    task = Task.model_validate(task_dict)
    env = get_environment()
    agent = LLMAgent(tools=env.get_tools(), domain_policy=env.get_policy(),
                     llm="hosted_vllm/qwen3.5-2B", llm_args=dict(AGENT_ARGS))
    user = UserSimulator(llm=customer, instructions=str(task.user_scenario), tools=None,
                         llm_args={"temperature": 0.0})
    orch = Orchestrator(domain="retail", agent=agent, user=user, environment=env, task=task,
                        max_steps=max_steps, max_errors=10, seed=seed)
    try:
        sim = run_simulation(orch)
    except Exception as e:  # noqa: BLE001 - an agent error is an outcome here, not a crash
        return {"task": task.id, "reward": None, "termination": f"exception:{type(e).__name__}",
                "agent_turns": None, "cost": 0.0}
    msgs = sim.messages
    cost = sum((getattr(m, "cost", 0.0) or 0.0) for m in msgs if m.role == "user")
    return {"task": task.id,
            "reward": sim.reward_info.reward if sim.reward_info else None,
            "termination": str(getattr(sim.termination_reason, "value", sim.termination_reason)),
            "agent_turns": sum(1 for m in msgs if m.role == "assistant"),
            "cost": cost}


def summarize(rows: list[dict]) -> dict:
    ok = [r for r in rows if r["reward"] is not None]
    turns = [r["agent_turns"] for r in ok if r["agent_turns"] is not None]
    return {"n": len(rows), "reward": round(sum(r["reward"] for r in ok) / len(ok), 4) if ok else None,
            "median_turns": statistics.median(turns) if turns else None,
            "terminations": dict(Counter(r["termination"] for r in rows))}


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("records")
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--trials", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=80)
    ap.add_argument("--customer", default="gpt-4.1-nano")
    ap.add_argument("--concurrency", type=int, default=64)
    a = ap.parse_args(argv[1:])
    _load_dotenv()
    if spent_usd() >= API_BUDGET_USD:
        raise SystemExit("API budget reached")

    ids = a.tasks.split(",")
    tasks = load_tasks()
    jobs = [(t, 7_000 + i) for t in ids for i in range(a.trials)]
    with ThreadPoolExecutor(a.concurrency) as pool:
        rows = list(pool.map(lambda j: run_tau2(tasks[j[0]], j[1], a.max_steps, a.customer), jobs))
    record_spend(sum(r["cost"] for r in rows), "fidelity tau2 orchestrator")

    # Driver side: tau2 names customer-ended runs user_stop; the driver calls them natural.
    rename = {"natural": "user_stop", "max_turns": "max_steps"}
    drv = [{"task": e.task_id, "reward": e.reward, "agent_turns": e.turn_count,
            "termination": rename.get(e.termination, e.termination)}
           for e in iter_episodes(a.records) if e.task_id in set(ids)]

    out = {"tau2": summarize(rows), "driver": summarize(drv),
           "per_task": {t: {"tau2": round(statistics.mean(r["reward"] for r in rows
                                                          if r["task"] == t and r["reward"] is not None), 3)
                            if any(r["task"] == t and r["reward"] is not None for r in rows) else None,
                            "driver": round(statistics.mean(r["reward"] for r in drv if r["task"] == t), 3)
                            if any(r["task"] == t for r in drv) else None}
                        for t in ids}}
    print(json.dumps(out, indent=1))
    (HERE / "runs" / "fidelity.json").write_text(json.dumps({"rows": rows, **out}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
