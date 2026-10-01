import glob, json, collections, os
from trainer import Config, Sampler, pass_k
from records import iter_episodes
adir = glob.glob("runs/p4-f/adapters/008-*")[0]
# prior: profile-v7 (+ p4-f's attempts) plus p4-g's step-0 batch, whose policy WAS p4-f step 8
prof = json.load(open("tasks/profile-v9.json"))
c = {"sampler": "value", "sampler_weight": "mixed", "sampler_pool": "all",
     "init_adapter": adir, "lr": 5e-6, "warmup_steps": 2, "kl_coef": 0.1,
     "unrequested_penalty": 0.5, "partial_credit": 0.3, "max_tokens_per_turn": 4096,
     "tool_hints_until": 0, "retire_n": 0, "steps": 12, "G": 16, "episodes_per_step": 512,
     "concurrency": 128, "tasks_file": "tb_train_v9.json", "split_file": "split_v9.json",
     "profile_file": "profile-v9.json", "val_every": 4, "val_G": 8, "val_repeat0": False}
json.dump(c, open("runs/p4-i.config.json", "w"), indent=1)
os.makedirs("runs/p4-i", exist_ok=True)
by = collections.defaultdict(list)
for e in iter_episodes("runs/valbig/p4f-008"):
    if not e.needs_reroll: by[e.task_id].append(e.eval_reward)
counts = [(sum(v), len(v)) for v in by.values()]
rec = {"step": 0, "repeat": 0, "policy_version": os.path.basename(adir), "tasks": len(counts),
       "episodes": sum(n for _, n in counts), "pass1": pass_k(counts, 1), "pass2": pass_k(counts, 2),
       "pass4": pass_k(counts, 4), "t_val_s": 0.0, "note": "seeded from runs/valbig/p4f-008"}
open("runs/p4-i/val.jsonl", "w").write(json.dumps(rec) + "\n")
cfg = Config(); [setattr(cfg, k, v) for k, v in c.items()]
sp = json.load(open("tasks/split_v9.json"))
band = [t for t in sp["train"] if t in prof and prof[t]["successes"] < prof[t]["trials"]
        and prof[t]["successes"] / prof[t]["trials"] >= 0.5]
json.dump({"train": sorted(band), "val": sp["val"], "all": sorted(band + sp["val"])}, open("tasks/split_v9b.json", "w"), indent=1)
c["split_file"] = "split_v9b.json"; json.dump(c, open("runs/p4-i.config.json", "w"), indent=1)
cfg.split_file = "split_v9b.json"
sm = Sampler(cfg, band, prof); ts, G = sm.next()
ps = [prof[t]["successes"] / prof[t]["trials"] for t in ts if t in prof]
print("pool", len(sm.pool), "| draw", len(ts), "x G", G, "| drawn tasks' mean measured p", round(sum(ps)/len(ps), 2),
      "| share with p in [0.6,0.95]", round(sum(0.6 <= p <= 0.95 for p in ps)/len(ps), 2),
      dict(collections.Counter("new t6c" if t.startswith("t6c") else "bundle" if "bundle" in t else "v6" if t.startswith("t6") else "hardened" for t in ts)))
