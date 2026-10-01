"""soupk = uniform average of four continuations of soup4: p4-j1..3 (step 4) and p4-k (step 6)."""
import glob, shutil
from pathlib import Path
from safetensors.torch import load_file, save_file
src = [sorted(glob.glob(f"runs/p4-j{i}/adapters/004-*"))[-1] for i in (1, 2, 3)]
src.append(sorted(glob.glob("runs/p4-k/adapters/006-*"))[-1])
out = Path("runs/soup/adapters/002-soupk"); out.mkdir(parents=True, exist_ok=True)
for sub in ["", "serve"]:
    ws = [load_file(str(Path(s) / sub / "adapter_model.safetensors")) for s in src]
    avg = {k: sum(w[k].float() for w in ws).div(len(ws)).to(ws[0][k].dtype) for k in ws[0]}
    (out / sub).mkdir(exist_ok=True)
    save_file(avg, str(out / sub / "adapter_model.safetensors"))
    shutil.copy(Path(src[0]) / sub / "adapter_config.json", out / sub / "adapter_config.json")
print("soupk from", src)
