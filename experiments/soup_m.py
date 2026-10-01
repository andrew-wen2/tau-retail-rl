"""soupm (Sept 29) = 50/50 average of soup4 and p4-m's adapter 003 (three updates at lr 4e-5; KL to
base 0.163 at it, before step 4's runaway rise). Halves p4-m's move away from soup4."""
import glob, shutil
from pathlib import Path
from safetensors.torch import load_file, save_file
src = ["runs/soup/adapters/000-soup4", sorted(glob.glob("runs/p4-m/adapters/003-*"))[-1]]
out = Path("runs/soup/adapters/003-soupm"); out.mkdir(parents=True, exist_ok=True)
for sub in ["", "serve"]:
    ws = [load_file(str(Path(s) / sub / "adapter_model.safetensors")) for s in src]
    avg = {k: sum(w[k].float() for w in ws).div(len(ws)).to(ws[0][k].dtype) for k in ws[0]}
    (out / sub).mkdir(exist_ok=True)
    save_file(avg, str(out / sub / "adapter_model.safetensors"))
    shutil.copy(Path(src[0]) / sub / "adapter_config.json", out / sub / "adapter_config.json")
print("soupm from", src)
