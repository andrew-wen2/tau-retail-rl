"""Configs for p4-j1..3 (Sept 28): three short branches from soup4 with persona-randomised
DeepSeek customers and tau3's reward rule (DB x COMMUNICATE), to be weight-averaged."""
import json, os
PERSONAS = {"forthcoming": 0.4, "terse": 0.3, "impatient": 0.15, "uncertain": 0.15}
for i, seed in enumerate((301, 302, 303), 1):
    c = {"sampler": "value", "sampler_weight": "mixed", "sampler_pool": "all",
         "init_adapter": "runs/soup/adapters/000-soup4", "lr": 1e-5, "warmup_steps": 1,
         "kl_coef": 0.05, "unrequested_penalty": 0.5, "partial_credit": 0.3,
         "max_tokens_per_turn": 4096, "tool_hints_until": 0, "retire_n": 0,
         "steps": 4, "G": 8, "episodes_per_step": 192, "concurrency": 128,
         "tasks_file": "tb_train_v10.json", "split_file": "split_v10.json",
         "profile_file": "profile-v10.json", "val_every": 0,
         "seed": seed, "val_seed": 31000899, "personas": PERSONAS}
    os.makedirs(f"runs/p4-j{i}", exist_ok=True)
    json.dump(c, open(f"runs/p4-j{i}.config.json", "w"), indent=1)
print("wrote runs/p4-j1..3.config.json")
