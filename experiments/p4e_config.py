import json, collections
c = {"sampler": "value", "sampler_weight": "mixed", "sampler_pool": "not_always",
     "init_adapter": "runs/p4-b/adapters/015-393fe2c97600", "lr": 1e-5, "warmup_steps": 2,
     "unrequested_penalty": 0.5, "partial_credit": 0.3, "kl_coef": 0.05,
     "max_tokens_per_turn": 4096, "tool_hints_until": 16,
     "family_adapt": True, "family_boost": 3.0, "promote_at": 0.85, "retire_n": 16, "retire_p": 0.93,
     "steps": 20, "G": 8, "episodes_per_step": 384, "concurrency": 128,
     "tasks_file": "tb_train_v5.json", "split_file": "split_v5.json",
     "profile_file": "profile-hard.json", "val_every": 8, "val_G": 8, "val_repeat0": False}
json.dump(c, open("runs/p4-e.config.json", "w"), indent=1)
from trainer import Config, Sampler
cfg = Config(); [setattr(cfg, k, v) for k, v in c.items()]
sm = Sampler(cfg, json.load(open("tasks/split_v5.json"))["train"], json.load(open("tasks/profile-hard.json")))
ts, _ = sm.next()
print("pool", len(sm.pool), "eligible", len(sm.eligible()), "| draw", len(ts), "tasks:",
      collections.Counter(t.split("_")[0] + ("_" + t[3:].rsplit("_", 2)[0] if t.startswith("tt_") else "") for t in ts))
