import json
c = {"sampler": "value", "sampler_weight": "mixed", "sampler_pool": "not_always",
     "init_adapter": "runs/p4-b/adapters/015-393fe2c97600", "unrequested_penalty": 0.5,
     "lr": 5e-6, "steps": 20, "G": 8, "episodes_per_step": 512, "concurrency": 128,
     "tasks_file": "tb_train_v4.json", "split_file": "split_v4.json",
     "profile_file": "profile-hard.json", "val_every": 4, "val_G": 8, "val_repeat0": False}
json.dump(c, open("runs/p4-d.config.json", "w"), indent=1)
from trainer import Config, Sampler
cfg = Config(); [setattr(cfg, k, v) for k, v in c.items()]
sm = Sampler(cfg, json.load(open("tasks/split_v4.json"))["train"], json.load(open("tasks/profile-hard.json")))
ts, _ = sm.next(); prof = json.load(open("tasks/profile-hard.json"))
print("pool", len(sm.pool), "| draw", len(ts), "tasks:", sum("_r" in t for t in ts), "variants,",
      sum(t.startswith("tg_") for t in ts), "generated,", sum(t in prof for t in ts), "profiled")
