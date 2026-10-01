"""Customer-transfer proxy (Sept 28): base vs the 4-way soup on valbig tasks that are not
solved 8/8 by both under DeepSeek, with the EVAL customer (gpt-5.2, low reasoning, tau2's own
instructions). 2 attempts per task, same seeds in both arms. Never touches the held-out 114."""
import collections, json, subprocess
from pathlib import Path
from trainer import HERE, TAU2_PYTHON, Config, swap_adapter
from records import iter_episodes
def succ(d):
    m = collections.defaultdict(list)
    for e in iter_episodes(HERE / d):
        if not e.needs_reroll: m[e.task_id].append(e.eval_reward)
    return m
b, s = succ("runs/valbig/base"), succ("runs/valbig/soup4")
tasks = sorted(t for t in b if t in s and (sum(b[t]) < len(b[t]) or sum(s[t]) < len(s[t])))
print("tasks", len(tasks), "of", len(b), flush=True)
m0 = json.load(open(HERE / "runs/p4-e/manifests/val0000.json"))
adir = HERE / "runs/soup/adapters/000-soup4"
arms = {"base": "qwen3.5-2B", "soup4": swap_adapter(Config(), adir, "px-soup4")}
procs = []
for i, (arm, model) in enumerate(arms.items()):
    m = dict(m0, batch_id=520000 + i, policy_version=arm, model=model, tasks=tasks, G=2,
             seed=52000001, out_dir=str(HERE / f"runs/proxy52/{arm}"), concurrency=64,
             customer_model="gpt-5.2", customer_args={"reasoning_effort": "low"},
             closing_instruction=False, guards={"max_steps": 200, "max_tokens_per_turn": 4096})
    mp = HERE / f"runs/proxy52/{arm}.json"; mp.parent.mkdir(parents=True, exist_ok=True)
    mp.write_text(json.dumps(m, indent=1))
    procs.append((arm, subprocess.Popen([str(TAU2_PYTHON), str(HERE / "driver.py"), "batch", str(mp)],
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=HERE)))
for arm, p in procs:
    out, err = p.communicate()
    print(arm, out.strip().splitlines()[-1][:500] if p.returncode == 0 else err[-1500:], flush=True)
