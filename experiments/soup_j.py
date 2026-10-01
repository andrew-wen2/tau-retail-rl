"""Uniform weight average of the three p4-j branch finals (step-4 adapters) -> runs/soup/adapters/001-soupj."""
import glob, shutil
from pathlib import Path
from safetensors.torch import load_file, save_file
src = [sorted(glob.glob(f"runs/p4-j{i}/adapters/004-*"))[-1] for i in (1, 2, 3)]
out = Path("runs/soup/adapters/001-soupj"); out.mkdir(parents=True, exist_ok=True)
for sub in ["", "serve"]:
    ws = [load_file(str(Path(s) / sub / "adapter_model.safetensors")) for s in src]
    avg = {k: sum(w[k].float() for w in ws).div(len(ws)).to(ws[0][k].dtype) for k in ws[0]}
    (out / sub).mkdir(exist_ok=True)
    save_file(avg, str(out / sub / "adapter_model.safetensors"))
    shutil.copy(Path(src[0]) / sub / "adapter_config.json", out / sub / "adapter_config.json")
print("soupj from", src)
