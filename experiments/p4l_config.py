"""Config for p4-l (Sept 29): ONE branch from soup4 with the research-round-2 changes:
  A. a customer mix, one per group: DeepSeek-V4-Flash at temperature 1.0 (was tau2's 0.0) and
     gpt-5-nano at low reasoning (the eval customer's family), 50/50. Personas off.
  B. lr 4e-5, 4x every earlier run: "LoRA Without Regret" puts LoRA's best RL lr ~10x full
     fine-tuning's, and every continuation plateaued at 1e-5 / 5e-6.
  +  RAFT's repeated-call penalty, 0.05 per repeat, capped at 0.2.
  D. the v12 pool: only tasks the soup4 lineage gets mixed (build_v12.py).
max_turns episodes (nano role-flip loops, 4 of 32 in the vet) get no advantage.
Everything else as p4-k (graded credit, mistake penalty, KL 0.05 to base, tau3 reward)."""
import json, os
c = {"sampler": "value", "sampler_weight": "mixed", "sampler_pool": "all",
     "init_adapter": "runs/soup/adapters/000-soup4", "lr": 4e-5, "warmup_steps": 1,
     "kl_coef": 0.05, "unrequested_penalty": 0.5, "partial_credit": 0.3,
     "credit_mode": "graded", "mistake_penalty": 0.1, "mistake_cap": 0.2,
     "repeat_penalty": 0.05, "repeat_cap": 0.2, "exclude_max_turns": True,
     "max_tokens_per_turn": 4096, "tool_hints_until": 0, "retire_n": 0,
     "steps": 6, "G": 8, "episodes_per_step": 192, "concurrency": 128,
     "tasks_file": "tb_train_v12.json", "split_file": "split_v12.json",
     "profile_file": "profile-v12.json", "val_every": 0,
     "seed": 305, "val_seed": 31000899,
     "customers": {
         "deepseek-t1": {"model": "deepinfra/deepseek-ai/DeepSeek-V4-Flash",
                         "args": {"temperature": 1.0}, "weight": 0.5},
         "gpt-5-nano": {"model": "gpt-5-nano", "args": {"reasoning_effort": "low"}, "weight": 0.5}}}
os.makedirs("runs/p4-l", exist_ok=True)
json.dump(c, open("runs/p4-l.config.json", "w"), indent=1)
print("wrote runs/p4-l.config.json")
