import random
from math import comb
from records import iter_episodes
def bytask(d):
    m = {}
    for e in iter_episodes(d):
        if not e.needs_reroll: m.setdefault(e.task_id, []).append(e.eval_reward)
    return m
pk = lambda xs, k: comb(int(sum(xs)), k) / comb(len(xs), k)
def cmp(name, a, b):
    ts = sorted(t for t in b if t in a and len(a[t]) >= 4 and len(b[t]) >= 4)
    out = [f"{name:<22} n={len(ts):3d}"]
    for k in (1, 2, 4):
        d = [pk(b[t], k) - pk(a[t], k) for t in ts]
        random.seed(k)
        bs = sorted(sum(random.choices(d, k=len(d))) / len(d) for _ in range(4000))
        ma = sum(pk(a[t], k) for t in ts) / len(ts); mb = ma + sum(d) / len(d)
        out.append(f"p^{k} {ma:.3f}->{mb:.3f} {mb-ma:+.3f} [{bs[100]:+.3f},{bs[3899]:+.3f}]")
    up = sum(sum(b[t])/len(b[t]) > sum(a[t])/len(a[t]) for t in ts); dn = sum(sum(b[t])/len(b[t]) < sum(a[t])/len(a[t]) for t in ts)
    print(" | ".join(out) + f" | up {up} down {dn}")
base = bytask("runs/valbig/base")
arms = [("p4-b step15", "runs/p4-e/val/v0000"), ("p4-f step8", "runs/valbig/p4f-008"),
        ("p4-g upd5 (006)", "runs/valbig/p4g-006"), ("p4-h step4", "runs/p4-h/val/v0004"),
        ("p4-i step4", "runs/p4-i/val/v0004"), ("soup of 4", "runs/valbig/soup4"), ("SFT from soup (2 ep)", "runs/valbig/sft-a-2")]
A = {n: bytask(d) for n, d in arms}
print("== vs untrained base (valbig 166x8)")
for n, _ in arms: cmp(n, base, A[n])
print("== vs p4-f step8")
for n, _ in arms:
    if n != "p4-f step8": cmp(n, A["p4-f step8"], A[n])
print("== vs soup of 4")
for n, _ in arms[-1:]: cmp(n, A["soup of 4"], A[n])
print("== val_hard 65x4 (p4-f run)")
cmp("p4-f s0->s8", bytask("runs/p4-f/val/v0000"), bytask("runs/p4-f/val/v0008"))
