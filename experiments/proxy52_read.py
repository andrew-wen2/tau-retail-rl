import random, collections
from records import iter_episodes
def succ(d):
    m = collections.defaultdict(list)
    for e in iter_episodes(d):
        if not e.needs_reroll: m[e.task_id].append(e.eval_reward)
    return m
def paired(a, b, label):
    ts = sorted(t for t in a if t in b and len(a[t]) == len(b[t]) >= 2)
    d = [sum(b[t])/len(b[t]) - sum(a[t])/len(a[t]) for t in ts]
    random.seed(0); bs = sorted(sum(random.choices(d, k=len(d)))/len(d) for _ in range(4000))
    ma = sum(sum(a[t])/len(a[t]) for t in ts)/len(ts)
    print(f"{label:<34} n={len(ts)} pass^1 {ma:.3f} -> {ma+sum(d)/len(d):.3f}  diff {sum(d)/len(d):+.3f} [{bs[100]:+.3f}, {bs[3899]:+.3f}]"
          f"  up {sum(x>0 for x in d)} down {sum(x<0 for x in d)}")
    return set(ts)
b52, s52 = succ("runs/proxy52/base"), succ("runs/proxy52/soup4")
ts = paired(b52, s52, "gpt-5.2 customer (2 att)")
bd, sd = succ("runs/valbig/base"), succ("runs/valbig/soup4")
bd = {t: v for t, v in bd.items() if t in ts}; sd = {t: v for t, v in sd.items() if t in ts}
paired(bd, sd, "DeepSeek customer, same tasks (8 att)")
