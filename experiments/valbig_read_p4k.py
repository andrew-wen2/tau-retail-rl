import sys; sys.path.insert(0, ".")  # run from the repo root as experiments/<script>
"""Read one adapter on the 166-task val (same seed and 4096 cap as p4-e's step-0 val)."""
import glob, json, subprocess, sys
from pathlib import Path
from trainer import HERE, TAU2_PYTHON, Config, swap_adapter
adir = Path(glob.glob(str(HERE / "runs/p4-k/adapters/006-*"))[0])
cfg = Config()
serving = swap_adapter(cfg, adir, "vb-p4k")
m = json.load(open(HERE / "runs/p4-e/manifests/val0000.json"))
m.update({"batch_id": 400501, "policy_version": adir.name, "model": serving,
          "out_dir": str(HERE / "runs/valbig/p4k")})
mp = HERE / "runs/valbig/p4k.json"; mp.write_text(json.dumps(m, indent=1))
print("seed", m["seed"], "guards", m["guards"], "tasks", len(m["tasks"]), "file", m["tasks_file"], flush=True)
p = subprocess.run([str(TAU2_PYTHON), str(HERE / "driver.py"), "batch", str(mp)], capture_output=True, text=True, cwd=HERE)
print(p.stdout.strip().splitlines()[-1][:400] if p.returncode == 0 else p.stderr[-2000:])
