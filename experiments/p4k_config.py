"""Config for p4-k (Sept 29): ONE branch from soup4 with graded write credit and a small
mistake penalty on successes, persona customers and the tau3 reward, on v11's refreshed prior.
Read alone and averaged with p4-j1..3 (four continuations of soup4, as soup4 itself was built)."""
import json, os
c = {"sampler": "value", "sampler_weight": "mixed", "sampler_pool": "all",
     "init_adapter": "runs/soup/adapters/000-soup4", "lr": 1e-5, "warmup_steps": 1,
     "kl_coef": 0.05, "unrequested_penalty": 0.5, "partial_credit": 0.3,
     "credit_mode": "graded", "mistake_penalty": 0.1, "mistake_cap": 0.2,
     "max_tokens_per_turn": 4096, "tool_hints_until": 0, "retire_n": 0,
     "steps": 6, "G": 8, "episodes_per_step": 192, "concurrency": 128,
     "tasks_file": "tb_train_v11.json", "split_file": "split_v11.json",
     "profile_file": "profile-v11.json", "val_every": 0,
     "seed": 304, "val_seed": 31000899,
     "personas": {"forthcoming": 0.4, "terse": 0.3, "impatient": 0.15, "uncertain": 0.15}}
os.makedirs("runs/p4-k", exist_ok=True)
json.dump(c, open("runs/p4-k.config.json", "w"), indent=1)
print("wrote runs/p4-k.config.json")
