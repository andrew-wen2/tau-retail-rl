"""gpt-5.2 customer-transfer check of soupm (Sept 29): one new arm on the proxy's 94 valbig tasks x 2,
same seeds, manifest and customer as runs/proxy52/{base,soup4,soupk}. Paired by proxy52m_read.py."""
import sys; sys.path.insert(0, ".")
import json, subprocess
from trainer import HERE, TAU2_PYTHON, Config, swap_adapter
adir = HERE / "runs/soup/adapters/003-soupm"
serving = swap_adapter(Config(), adir, "px-soupm")
m = json.load(open(HERE / "runs/proxy52/soup4.json"))
m.update(batch_id=520003, policy_version="soupm", model=serving, out_dir=str(HERE / "runs/proxy52/soupm"))
mp = HERE / "runs/proxy52/soupm.json"; mp.write_text(json.dumps(m, indent=1))
p = subprocess.run([str(TAU2_PYTHON), str(HERE / "driver.py"), "batch", str(mp)], capture_output=True, text=True, cwd=HERE)
print(p.stdout.strip().splitlines()[-1][:500] if p.returncode == 0 else p.stderr[-2000:])
