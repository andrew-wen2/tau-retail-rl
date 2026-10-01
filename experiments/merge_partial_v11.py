"""The v11 light profile was stopped at the user's request (Sept 29) after ~half its tasks;
merge the complete groups it wrote into the prior (adding, as PROFILE_MERGE=add would) and
write profile-v11.json so p4k.sh skips the profile step."""
import sys; sys.path.insert(0, ".")
import collections, json
from records import iter_episodes
prior = json.load(open("tasks/profile-v11-prior.json"))
todo = json.load(open("tasks/to_profile_v11.json"))
by = collections.defaultdict(list)
for e in iter_episodes("runs/profile-v11"):
    if not e.needs_reroll:
        by[e.task_id].append(e.eval_reward)
done = {t: v for t, v in by.items() if len(v) >= 4}
for t, v in done.items():
    p = prior.setdefault(t, {"successes": 0, "trials": 0})
    p["successes"] += sum(v); p["trials"] += len(v)
json.dump(prior, open("tasks/profile-v11.json", "w"), indent=1)
ps = [sum(v) / len(v) for v in done.values()]
print(f"profiled {len(done)} of {len(todo)} planned tasks ({sum(map(len, done.values()))} episodes) | "
      f"new attempts: always {sum(p == 1 for p in ps)/len(ps):.0%}, mixed {sum(0 < p < 1 for p in ps)/len(ps):.0%}, "
      f"never {sum(p == 0 for p in ps)/len(ps):.0%}, mean {sum(ps)/len(ps):.2f}")
