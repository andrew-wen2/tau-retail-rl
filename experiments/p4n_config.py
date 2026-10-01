"""Config for p4-n (Sept 29): p4-m at lr 2e-5. p4-m (4e-5) roughly doubled KL to base every step
(0.015 -> 0.296 by step 4) and turned unstable from step 3 (grad norm 0.07 -> 0.36, length cuts
1 -> 17), yet soupm (soup4 + its step-3 adapter) gave the best gpt-5.2 read so far (+0.053 vs
base, n.s.). Same customers (DeepSeek T=1 + gpt-5-mini), pool, penalties and seed as p4-m, so
step 0 draws the same tasks."""
import json, os
c = json.load(open("runs/p4-m.config.json"))
c.update(lr=2e-5)
os.makedirs("runs/p4-n", exist_ok=True)
json.dump(c, open("runs/p4-n.config.json", "w"), indent=1)
print("wrote runs/p4-n.config.json", c["lr"], list(c["customers"]))
