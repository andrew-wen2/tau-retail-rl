"""Does the trained arm's gain over base depend on the customer? Same tasks, two customers.

For each customer the gain is trained - base, paired by task. This prints both gains, their
difference (gain A - gain B), and how each arm's score moves between the two customers, all with a
paired bootstrap over tasks. Trial counts may differ between customers; a task enters pass^k only
if all four runs have at least k scored trials for it.

Usage: python paired_customers.py BASE_A TRAINED_A BASE_B TRAINED_B [B=10000]"""
import json, random, sys
from collections import defaultdict
from math import comb


def load(path):
    out = defaultdict(list)
    for s in json.load(open(path))["simulations"]:
        if s.get("reward_info"):
            out[s["task_id"]].append(s["reward_info"]["reward"] >= 1.0)
    return out


def pass_k(trials, k):
    return comb(sum(trials), k) / comb(len(trials), k)


def main():
    ba, ta, bb, tb = (load(p) for p in sys.argv[1:5])
    B = int(sys.argv[5]) if len(sys.argv) > 5 else 10000
    for k in (1, 2, 4):
        tasks = sorted(x for x in ba if all(len(r[x]) >= k for r in (ba, ta, bb, tb)))
        rng = random.Random(300)
        idx = [[rng.randrange(len(tasks)) for _ in tasks] for _ in range(B)]
        p = {n: [pass_k(r[x], k) for x in tasks] for n, r in (("ba", ba), ("ta", ta), ("bb", bb), ("tb", tb))}
        rows = [("gain A (trained - base)", [y - x for x, y in zip(p["ba"], p["ta"])]),
                ("gain B (trained - base)", [y - x for x, y in zip(p["bb"], p["tb"])]),
                ("gain A - gain B", [(ya - xa) - (yb - xb) for xa, ya, xb, yb in zip(p["ba"], p["ta"], p["bb"], p["tb"])]),
                ("base: A - B", [x - y for x, y in zip(p["ba"], p["bb"])]),
                ("trained: A - B", [x - y for x, y in zip(p["ta"], p["tb"])])]
        print(f"== pass^{k} ({len(tasks)} tasks)  base A {100 * sum(p['ba']) / len(tasks):.1f}  trained A "
              f"{100 * sum(p['ta']) / len(tasks):.1f}  base B {100 * sum(p['bb']) / len(tasks):.1f}  trained B {100 * sum(p['tb']) / len(tasks):.1f}")
        for name, d in rows:
            bs = sorted(100 * sum(d[i] for i in ix) / len(ix) for ix in idx)
            print(f"{name:24s} {100 * sum(d) / len(d):+5.1f}  95% CI [{bs[int(0.025 * B)]:+5.1f}, {bs[int(0.975 * B)]:+5.1f}]  "
                  f"P(<=0) {sum(v <= 0 for v in bs) / B:.3f}")


main()
