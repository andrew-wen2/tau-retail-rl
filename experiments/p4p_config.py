"""Config for p4-p (Sept 30): p4-o (lr 3e-5) with two stabilizers instead of a lower lr.
p4-n (2e-5) never moved off soup4 (KL to base ~0.01 for 6 steps); p4-o (3e-5) and p4-m (4e-5)
drifted within 2-4 steps (KL to base 0.131 / 0.296, runaway generations 13 / 17 of 192).
  1. KL anchored to soup4 itself (kl_ref "init", coef 0.1), not to base: the penalty starts at 0
     and bounds the move away from soup4.
  2. Skip the update when > 3% of the batch hit the token cap (p4-o's step 0 drew 8 of 192 from
     soup4 and its update started the drift).
Same customers, pool, penalties and seed as p4-m/n/o."""
import json, os
c = json.load(open("runs/p4-m.config.json"))
c.update(lr=3e-5, kl_coef=0.1, kl_ref="init", max_runaway_frac=0.03)
os.makedirs("runs/p4-p", exist_ok=True)
json.dump(c, open("runs/p4-p.config.json", "w"), indent=1)
print("wrote runs/p4-p.config.json", {k: c[k] for k in ("lr", "kl_coef", "kl_ref", "max_runaway_frac")})
