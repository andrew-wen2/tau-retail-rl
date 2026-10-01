"""gpt-5.2 customer-transfer check of p4-p's final adapter (Sept 30): one new arm on the proxy's 94
valbig tasks x 2, same seeds, manifest and customer as runs/proxy52/{base,soup4,soupk,soupm}."""
import sys; sys.path.insert(0, ".")
import glob, json, subprocess
from pathlib import Path
from trainer import HERE, TAU2_PYTHON, Config, swap_adapter
adir = Path(sorted(glob.glob(str(HERE / "runs/p4-p/adapters/006-*")))[-1])
serving = swap_adapter(Config(), adir, "px-p4p")
m = json.load(open(HERE / "runs/proxy52/soup4.json"))
m.update(batch_id=520004, policy_version="p4p-006", model=serving, out_dir=str(HERE / "runs/proxy52/p4p"))
mp = HERE / "runs/proxy52/p4p.json"; mp.write_text(json.dumps(m, indent=1))
p = subprocess.run([str(TAU2_PYTHON), str(HERE / "driver.py"), "batch", str(mp)], capture_output=True, text=True, cwd=HERE)
print("adapter", adir.name)
print(p.stdout.strip().splitlines()[-1][:500] if p.returncode == 0 else p.stderr[-2000:])
