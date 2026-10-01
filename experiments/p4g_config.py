import glob, json, collections, os
from trainer import Config, Sampler, pass_k
from records import iter_episodes
adir = glob.glob("runs/p4-f/adapters/008-*")[0]
c = {"sampler": "value", "sampler_weight": "mixed", "sampler_pool": "not_always",
     "init_adapter": adir, "lr": 1e-5, "warmup_steps": 2,
     "unrequested_penalty": 0.5, "partial_credit": 0.3, "kl_coef": 0.05, "max_tokens_per_turn": 4096,
     "tool_hints_until": 8, "retire_n": 16, "steps": 16, "G": 8, "episodes_per_step": 256,
     "concurrency": 128, "tasks_file": "tb_train_v7.json", "split_file": "split_v7.json",
     "profile_file": "profile-v7.json", "val_every": 8, "val_G": 8, "val_repeat0": False}
json.dump(c, open("runs/p4-g.config.json", "w"), indent=1)
# step-0 val = the p4-f step-8 read on the same 166 tasks, seed and cap: seed it so it is not re-run
os.makedirs("runs/p4-g", exist_ok=True)
by = collections.defaultdict(list)
for e in iter_episodes("runs/valbig/p4f-008"):
    if not e.needs_reroll: by[e.task_id].append(e.eval_reward)
counts = [(sum(v), len(v)) for v in by.values()]
rec = {"step": 0, "repeat": 0, "policy_version": os.path.basename(adir), "tasks": len(counts),
       "episodes": sum(n for _, n in counts), "pass1": pass_k(counts, 1), "pass2": pass_k(counts, 2),
       "pass4": pass_k(counts, 4), "t_val_s": 0.0, "note": "seeded from runs/valbig/p4f-008"}
open("runs/p4-g/val.jsonl", "w").write(json.dumps(rec) + "\n")
cfg = Config(); [setattr(cfg, k, v) for k, v in c.items()]
sp = json.load(open("tasks/split_v7.json")); prof = json.load(open("tasks/profile-v7.json"))
sm = Sampler(cfg, sp["train"], prof); ts, _ = sm.next()
print("val step0 seeded", round(rec["pass1"], 3), "| train", len(sp["train"]), "| pool", len(sm.pool),
      "| first draw", dict(collections.Counter(t.split("_")[0] + ("_" + t.split("_")[1] if t.startswith("t6") else "") for t in ts)))
