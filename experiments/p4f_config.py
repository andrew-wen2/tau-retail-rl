import json, collections
tasks = {t["id"] for t in json.load(open("tasks/tb_train_v6.json"))}
sp = json.load(open("tasks/split_v6.json")); prof = json.load(open("tasks/profile-v6.json"))
# Train only on tasks the sampler has real counts for -- the new v6 tasks
# (profiled with the starting adapter) and the original hardened set (profile-hard); variants and
# gen_targeted's tasks are unprofiled for this adapter and were p4-e's all-success waste
train = [t for t in sp["train"] if t in prof and (t.startswith("t6_") or (t.startswith(("tb_", "tbc_")) and "_r" not in t))]
json.dump({"train": sorted(train), "val": sp["val_hard"], "all": sorted(train + sp["val_hard"])},
          open("tasks/split_v6f.json", "w"), indent=1)
c = {"sampler": "value", "sampler_weight": "mixed", "sampler_pool": "not_always",
     "init_adapter": "runs/p4-b/adapters/015-393fe2c97600", "lr": 1e-5, "warmup_steps": 2,
     "unrequested_penalty": 0.5, "partial_credit": 0.3, "kl_coef": 0.05, "max_tokens_per_turn": 4096,
     "tool_hints_until": 6, "retire_n": 16, "steps": 8, "G": 8, "episodes_per_step": 256,
     "concurrency": 128, "tasks_file": "tb_train_v6.json", "split_file": "split_v6f.json",
     "profile_file": "profile-v6.json", "val_every": 8, "val_G": 4, "val_repeat0": False}
json.dump(c, open("runs/p4-f.config.json", "w"), indent=1)
from trainer import Config, Sampler
cfg = Config(); [setattr(cfg, k, v) for k, v in c.items()]
sm = Sampler(cfg, train, prof)
ts, _ = sm.next()
print("train", len(train), "| pool (not always solved)", len(sm.pool), "| first draw", len(ts),
      dict(collections.Counter(t.split("_")[1] if t.startswith("t6_") else "hardened" for t in ts)))
