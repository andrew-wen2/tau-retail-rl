import glob, json, collections, os
from trainer import Config, Sampler, pass_k
from records import iter_episodes
adir = glob.glob("runs/p4-f/adapters/008-*")[0]
# prior: profile-v7 (+ p4-f's attempts) plus p4-g's step-0 batch, whose policy WAS p4-f step 8
prof = json.load(open("tasks/profile-v7.json"))
for e in iter_episodes("runs/p4-g/records/b0000"):
    if not e.needs_reroll and e.task_id in prof:
        prof[e.task_id]["successes"] += e.eval_reward; prof[e.task_id]["trials"] += 1
json.dump(prof, open("tasks/profile-v8.json", "w"), indent=1)
c = {"sampler": "value", "sampler_weight": "pass4", "sampler_pool": "all",
     "init_adapter": adir, "lr": 5e-6, "warmup_steps": 2, "kl_coef": 0.2,
     "unrequested_penalty": 0.5, "partial_credit": 0.3, "max_tokens_per_turn": 4096,
     "tool_hints_until": 0, "retire_n": 0, "steps": 12, "G": 16, "episodes_per_step": 512,
     "concurrency": 128, "tasks_file": "tb_train_v7.json", "split_file": "split_v7.json",
     "profile_file": "profile-v8.json", "val_every": 4, "val_G": 8, "val_repeat0": False}
json.dump(c, open("runs/p4-h.config.json", "w"), indent=1)
os.makedirs("runs/p4-h", exist_ok=True)
by = collections.defaultdict(list)
for e in iter_episodes("runs/valbig/p4f-008"):
    if not e.needs_reroll: by[e.task_id].append(e.eval_reward)
counts = [(sum(v), len(v)) for v in by.values()]
rec = {"step": 0, "repeat": 0, "policy_version": os.path.basename(adir), "tasks": len(counts),
       "episodes": sum(n for _, n in counts), "pass1": pass_k(counts, 1), "pass2": pass_k(counts, 2),
       "pass4": pass_k(counts, 4), "t_val_s": 0.0, "note": "seeded from runs/valbig/p4f-008"}
open("runs/p4-h/val.jsonl", "w").write(json.dumps(rec) + "\n")
cfg = Config(); [setattr(cfg, k, v) for k, v in c.items()]
sp = json.load(open("tasks/split_v7.json"))
sm = Sampler(cfg, sp["train"], prof); ts, G = sm.next()
ps = [prof[t]["successes"] / prof[t]["trials"] for t in ts if t in prof]
print("pool", len(sm.pool), "| draw", len(ts), "x G", G, "| drawn tasks' mean measured p", round(sum(ps)/len(ps), 2),
      "| share with p in [0.6,0.95]", round(sum(0.6 <= p <= 0.95 for p in ps)/len(ps), 2),
      dict(collections.Counter("bundle" if "bundle" in t else "v6" if t.startswith("t6") else "hardened" for t in ts)))
