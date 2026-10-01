"""Step 1 of the SFT build (tau2 venv): episodes -> chat messages exactly as the driver built
them for the agent, plus the system prompt and tool schemas (the same for every retail task).
Teacher successes are the fine-tuning data; 2B successes are (a) the format check -- their
recorded token ids must equal our re-render -- and (b) the replay mix against forgetting."""
import glob, json, random
from driver import GREETING, Tau2Retail, load_tasks
from records import AssistantTurn, iter_episodes
tasks = load_tasks(__import__("pathlib").Path("tasks/tb_train_v9.json"))
rt = Tau2Retail(next(iter(tasks.values())))
system, tools = rt.system_prompt(), rt.tool_schemas()

def messages(ep):
    msgs = [{"role": "system", "content": system},
            {"role": "assistant", "content": GREETING, "tool_calls": None}]
    pending = []
    for t in ep.turns:
        if isinstance(t, AssistantTurn):
            calls = t.tool_calls or []
            pending = [c["id"] for c in calls]
            msgs.append({"role": "assistant", "reasoning_content": "",
                         "content": t.text if (t.text or not calls) else None,
                         "tool_calls": [{"id": c["id"], "type": "function",
                                         "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])}}
                                        for c in calls] or None})
        elif t.role == "tool":
            msgs.append({"role": "tool", "content": t.content, "tool_call_id": pending.pop(0)})
        elif t.role == "user":
            msgs.append({"role": "user", "content": t.content})
    return msgs

AUTH = {"find_user_id_by_email", "find_user_id_by_name_zip", "get_user_details"}
WRITE_PREFIX = ("cancel_", "modify_", "return_", "exchange_")
def auth_first(ep):
    for t in ep.assistant_turns:
        for c in t.tool_calls or []:
            if c["name"] in AUTH: return True
            if c["name"].startswith(WRITE_PREFIX): return False
    return True

unsolvable = set(json.load(open("tasks/unsolvable_tasks.json")))
out = {"system": system, "tools": tools, "teacher": [], "own": [], "check": []}
drop = {"not_auth_first": 0}
for ep in iter_episodes("runs/teacher/v41f"):
    if ep.eval_reward != 1: continue
    if not auth_first(ep): drop["not_auth_first"] += 1; continue
    out["teacher"].append({"task_id": ep.task_id, "episode_id": ep.episode_id, "messages": messages(ep)})
own = {}
for d in sorted(glob.glob("runs/p4-i/records/b*")):
    for ep in iter_episodes(d):
        if ep.eval_reward == 1 and ep.trains and ep.task_id not in unsolvable and auth_first(ep):
            own.setdefault(ep.task_id, []).append(ep)
random.seed(7)
picked = [random.choice(v) for v in own.values()]
random.shuffle(picked)
for ep in picked[:len(out["teacher"])]:
    a = ep.assistant_turns
    out["own"].append({"task_id": ep.task_id, "episode_id": ep.episode_id,
                       "turns": [[t.prompt_token_ids, t.token_ids] for t in a]})
for ep in picked[-20:]:
    a = ep.assistant_turns
    out["check"].append({"episode_id": ep.episode_id, "messages": messages(ep),
                         "turns": [[t.prompt_token_ids, t.token_ids] for t in a]})
json.dump(out, open("sft_raw.json", "w"))
print("teacher", len(out["teacher"]), "own", len(out["own"]), "of", len(own), "tasks with own successes",
      "| check", len(out["check"]), "| dropped", drop)
