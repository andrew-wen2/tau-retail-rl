"""Uniform weight average ("soup") of four continuations of p4-f step 8, as one rank-32 LoRA.
A and B are averaged separately; the error vs. the exact average of the B@A deltas is reported."""
import glob, shutil, torch
from pathlib import Path
from safetensors.torch import load_file, save_file
src = [glob.glob(p)[0] for p in ["runs/p4-f/adapters/008-*", "runs/p4-g/adapters/006-*",
                                 "runs/p4-h/adapters/004-*", "runs/p4-i/adapters/004-*"]]
out = Path("runs/soup/adapters/000-soup4"); out.mkdir(parents=True, exist_ok=True)
for sub in ["", "serve"]:
    ws = [load_file(f"{s}/{sub}/adapter_model.safetensors".replace("//", "/")) for s in src]
    avg = {k: sum(w[k].float() for w in ws).div(len(ws)).to(ws[0][k].dtype) for k in ws[0]}
    (out / sub).mkdir(exist_ok=True)
    save_file(avg, str(out / sub / "adapter_model.safetensors"))
    shutil.copy(f"{src[0]}/{sub}/adapter_config.json".replace("//", "/"), out / sub / "adapter_config.json")
    if sub == "serve":
        num = den = drift = 0.0
        for ka in [k for k in avg if "lora_A" in k]:
            kb = ka.replace("lora_A", "lora_B")
            exact = sum(w[kb].float() @ w[ka].float() for w in ws) / len(ws)
            approx = avg[kb].float() @ avg[ka].float()
            f = ws[0][kb].float() @ ws[0][ka].float()
            num += (approx - exact).norm() ** 2; den += exact.norm() ** 2; drift += (exact - f).norm() ** 2
        print(f"relative error of averaged-A/B vs exact delta average: {(num/den)**.5:.4f}; "
              f"soup's distance from p4-f step 8 (relative): {(drift/den)**.5:.4f}")
print("wrote", out, "from", [Path(s).parent.parent.name + "/" + Path(s).name for s in src])
