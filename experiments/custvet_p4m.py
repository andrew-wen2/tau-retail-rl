"""Pre-flight for p4-m (Sept 29): pick the OpenAI half of the customer mix. p4-l's step 0 showed
gpt-5-nano failing on the multi-request tasks that dominate the v12 pool (0.302 success, 17 of
96 at the step cap, role-flips into the agent) while its 16-task vet on mostly single tasks had
passed. So this vet uses 16 BUNDLE tasks from the pool x 2, soup4, same seeds for every arm, with
DeepSeek T=1 as the reference. A candidate passes if it loops (max_turns) at most 2 of 32 times,
succeeds no worse than DeepSeek - 0.10, and costs <= MAX_USD_PER_EP. Writes choice.json: the
first passing candidate in preference order (gpt-5-mini: the eval customer's family)."""
import sys; sys.path.insert(0, ".")
import json, random, statistics, subprocess
from collections import Counter
from trainer import HERE, TAU2_PYTHON, Config, swap_adapter
from records import iter_episodes
MAX_USD_PER_EP = 0.02
ARMS = {"deepseek-t1": {"model": "deepinfra/deepseek-ai/DeepSeek-V4-Flash", "args": {"temperature": 1.0}},
        "gpt-5-mini": {"model": "gpt-5-mini", "args": {"reasoning_effort": "low"}},
        "gpt-4.1-mini": {"model": "gpt-4.1-mini", "args": {"temperature": 1.0}}}
pool = json.load(open(HERE / "tasks" / "split_v12.json"))["train"]
tasks = random.Random(306).sample(sorted(t for t in pool if "bundle" in t), 16)
serving = swap_adapter(Config(), HERE / "runs/soup/adapters/000-soup4", "vetm-soup4")
root = HERE / "runs/custvet_p4m"
res = {}
for i, (name, spec) in enumerate(ARMS.items()):
    out = root / name
    if not list(out.glob("_batch*.summary.json")):
        m = {"batch_id": 540000 + i, "policy_version": "soup4", "model": serving, "tasks": tasks, "G": 2,
             "seed": 5401, "out_dir": str(out), "concurrency": 32,
             "customer_model": spec["model"], "customer_args": spec["args"],
             "guards": {"max_steps": 80, "max_tokens_per_turn": 4096}, "tasks_file": "tb_train_v12.json"}
        mp = out.with_suffix(".json"); mp.parent.mkdir(parents=True, exist_ok=True); mp.write_text(json.dumps(m))
        p = subprocess.run([str(TAU2_PYTHON), str(HERE / "driver.py"), "batch", str(mp)],
                           capture_output=True, text=True, cwd=HERE)
        if p.returncode:
            print(f"  {name}: driver failed\n{p.stderr[-1500:]}"); continue
    s = json.loads(next(out.glob("_batch*.summary.json")).read_text())
    eps = [e for e in iter_episodes(out) if not e.needs_reroll]
    term = Counter(e.termination for e in eps)
    succ = sum(e.eval_reward for e in eps) / max(1, len(eps))
    usd = s["customer_usd"] / max(1, len(eps))
    res[name] = {"n": len(eps), "success": succ, "loops": term.get("max_turns", 0), "usd": usd}
    print(f"  {name:12s} n={len(eps)} success {succ:.3f} terminations {dict(term)} ${usd:.5f}/ep "
          f"median wall {statistics.median(e.wall_clock_s for e in eps) if eps else 0:.0f}s")
ref = res.get("deepseek-t1", {}).get("success")
choice = None
for name in ("gpt-5-mini", "gpt-4.1-mini"):
    r = res.get(name)
    why = ("no result" if not r or r["n"] < 24 else f"{r['loops']} loops" if r["loops"] > 2
           else f"success {r['success']:.3f} < DeepSeek {ref:.3f} - 0.10" if ref is not None and r["success"] < ref - 0.10
           else f"${r['usd']:.4f}/ep" if r["usd"] > MAX_USD_PER_EP else None)
    print(f"  {name}: {'PASS' if why is None else 'fail: ' + why}")
    if why is None and choice is None:
        choice = name
if choice is None:
    print("VET FAILED: no OpenAI customer passed"); sys.exit(1)
(root / "choice.json").write_text(json.dumps({"name": choice, **ARMS[choice], "vet": res}, indent=1))
print(f"VET-OK chose {choice}")
