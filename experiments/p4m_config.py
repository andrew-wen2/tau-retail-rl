"""Config for p4-m (Sept 29): p4-l with the OpenAI customer replaced. p4-l's gpt-5-nano failed on
multi-request tasks (0.302 vs DeepSeek 0.677 at step 0, 17 of 96 step-cap loops, role-flips);
it was stopped after step 0 and its update is not used. p4-m restarts from soup4 with the customer
custvet_p4m.py chose (gpt-5-mini low reasoning, else gpt-4.1-mini T=1) 50/50 with DeepSeek T=1.
Everything else as p4-l: lr 4e-5, repeat penalty 0.05 capped 0.2, v12 pool, max_turns episodes
get no advantage, graded credit, mistake penalty, KL 0.05 to base, tau3 reward."""
import json, os
ch = json.load(open("runs/custvet_p4m/choice.json"))
c = json.load(open("runs/p4-l.config.json"))
c.update(seed=306, customers={
    "deepseek-t1": {"model": "deepinfra/deepseek-ai/DeepSeek-V4-Flash", "args": {"temperature": 1.0}, "weight": 0.5},
    ch["name"]: {"model": ch["model"], "args": ch["args"], "weight": 0.5}})
os.makedirs("runs/p4-m", exist_ok=True)
json.dump(c, open("runs/p4-m.config.json", "w"), indent=1)
print(f"wrote runs/p4-m.config.json with {ch['name']}")
