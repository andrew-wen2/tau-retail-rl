"""Pre-flight for p4-l's customer mix (Sept 29): soup4 on 16 v12-pool tasks x 2 per customer,
same seeds both arms. Fails (exit 1) if a customer errors, loops to the step cap, or costs more
than MAX_USD_PER_EP, so a broken customer never reaches a training step. ~64 episodes."""
import sys; sys.path.insert(0, ".")
import json, random, statistics, subprocess
from collections import Counter
from trainer import HERE, TAU2_PYTHON, Config, swap_adapter
from records import iter_episodes
MAX_USD_PER_EP = 0.004
cfg = json.load(open(HERE / "runs/p4-l.config.json"))
pool = json.load(open(HERE / "tasks" / "split_v12.json"))["train"]
tasks = random.Random(305).sample(pool, 16)
serving = swap_adapter(Config(), HERE / "runs/soup/adapters/000-soup4", "vet-soup4")
bad = []
for i, (name, spec) in enumerate(cfg["customers"].items()):
    out = HERE / "runs/custvet_p4l" / name
    m = {"batch_id": 530000 + i, "policy_version": "soup4", "model": serving, "tasks": tasks, "G": 2,
         "seed": 5301, "out_dir": str(out), "concurrency": 32,
         "customer_model": spec["model"], "customer_args": spec["args"],
         "guards": {"max_steps": 80, "max_tokens_per_turn": 4096}, "tasks_file": cfg["tasks_file"]}
    mp = out.with_suffix(".json"); mp.parent.mkdir(parents=True, exist_ok=True); mp.write_text(json.dumps(m))
    p = subprocess.run([str(TAU2_PYTHON), str(HERE / "driver.py"), "batch", str(mp)],
                       capture_output=True, text=True, cwd=HERE)
    if p.returncode:
        print(p.stderr[-2000:]); bad.append(f"{name}: driver failed"); continue
    s = json.loads(p.stdout.strip().splitlines()[-1])
    eps = [e for e in iter_episodes(out) if not e.needs_reroll]
    term = Counter(e.termination for e in eps)
    usd = s["customer_usd"] / max(1, len(eps))
    wall = statistics.median(e.wall_clock_s for e in eps) if eps else 0
    print(f"  {name:12s} n={len(eps)} success {sum(e.eval_reward for e in eps) / max(1, len(eps)):.3f} "
          f"terminations {dict(term)} ${usd:.5f}/ep median wall {wall:.0f}s "
          f"turns {statistics.median(e.turn_count for e in eps) if eps else 0}")
    if len(eps) < 24: bad.append(f"{name}: only {len(eps)} of 32 episodes")
    if term.get("max_turns", 0) > 6: bad.append(f"{name}: {term['max_turns']} step-cap loops")
    if usd > MAX_USD_PER_EP: bad.append(f"{name}: ${usd:.4f}/ep")
print("VET FAILED: " + "; ".join(bad) if bad else "VET-OK")
sys.exit(1 if bad else 0)
