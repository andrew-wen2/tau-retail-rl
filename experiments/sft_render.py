"""Step 2 of the SFT build (train venv): render teacher conversations with the SERVED template
(baseline/qwen35_train.jinja, thinking off) into token ids + a loss mask on agent tokens only.
The renderer is first checked against 20 recorded 2B episodes: every agent turn's prompt and
generated tokens must come out identical to what vLLM recorded, or nothing is written."""
import json, sys
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-2B")
tmpl = open("baseline/qwen35_train.jinja").read()
raw = json.load(open("sft_raw.json"))
tools = raw["tools"]
def _obj_args(msgs):
    # vLLM's server parses tool-call arguments into a mapping before templating; do the same
    out = []
    for m in msgs:
        if m.get("tool_calls"):
            m = dict(m, tool_calls=[dict(c, function=dict(c["function"], arguments=json.loads(c["function"]["arguments"])))
                                    for c in m["tool_calls"]])
        out.append(m)
    return out
def render(msgs, gen):
    msgs = _obj_args(msgs)
    return tok.apply_chat_template(msgs, tools=tools, chat_template=tmpl, tokenize=True,
                                   add_generation_prompt=gen, enable_thinking=False,
                                   return_dict=False)
END = tok.convert_tokens_to_ids("<|im_end|>")
def spans(msgs):
    """[(prompt_ids, completion_ids)] per agent turn, completion ending at <|im_end|>."""
    out = []
    for i, m in enumerate(msgs):
        if m["role"] != "assistant" or i < 2: continue
        p = render(msgs[:i], True); f = render(msgs[:i + 1], False)
        if f[:len(p)] != p: raise ValueError(f"prefix broken at message {i}")
        c = f[len(p):]
        c = c[:c.index(END) + 1]
        out.append((p, c))
    return out
bad = 0
for ex in raw["check"]:
    got = spans(ex["messages"])
    ok = len(got) == len(ex["turns"]) and all(list(p) == pr and list(c) == co
                                               for (p, c), (pr, co) in zip(got, ex["turns"]))
    if not ok:
        bad += 1
        for k, ((p, c), (pr, co)) in enumerate(zip(got, ex["turns"])):
            if list(p) != pr or list(c) != co:
                j = next((x for x in range(min(len(p), len(pr))) if p[x] != pr[x]), min(len(p), len(pr)))
                print("MISMATCH", ex["episode_id"], "turn", k, "prompt_eq", list(p) == pr, "compl_eq", list(c) == co,
                      "| at", j, repr(tok.decode(p[max(0, j-20):j+20])), "vs", repr(tok.decode(pr[max(0, j-20):j+20])),
                      "| compl", repr(tok.decode(c[:60])), "vs", repr(tok.decode(co[:60])))
                break
print(f"format check: {len(raw['check']) - bad}/{len(raw['check'])} recorded 2B episodes reproduced exactly")
if bad: sys.exit(1)
def example(turns, source, tid):
    ids, mask = list(turns[-1][0]) + list(turns[-1][1]), None
    mask = [0] * len(ids)
    for p, c in turns:
        if ids[:len(p)] != list(p): raise ValueError("turn prompt is not a prefix of the final sequence")
        for x in range(len(p), len(p) + len(c)): mask[x] = 1
    return {"task_id": tid, "source": source, "input_ids": ids, "loss_mask": mask}
data, lens = [], []
for ex in raw["teacher"]:
    data.append(example(spans(ex["messages"]), "teacher", ex["task_id"]))
for ex in raw["own"]:
    data.append(example(ex["turns"], "own", ex["task_id"]))
for d in data: lens.append((d["source"], len(d["input_ids"]), sum(d["loss_mask"])))
json.dump(data, open("sft_v1.json", "w"))
for s in ("teacher", "own"):
    L = sorted(x[1] for x in lens if x[0] == s); T = sorted(x[2] for x in lens if x[0] == s)
    print(f"{s}: {len(L)} examples | seq len median {L[len(L)//2]} max {L[-1]} | trained tokens median {T[len(T)//2]} total {sum(T)}")
