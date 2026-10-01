#!/usr/bin/env bash
# Every check that should pass before a training run, in dependency order. Runs to the end even
# when a check fails and prints a PASS/FAIL table. Run from the repo root (the VM home), detached:
#   setsid nohup baseline/pretrain_tests.sh > pretrain_tests.log 2>&1 < /dev/null &
#
#   1 env        check_train_env.py: pinned versions, fast DeltaNet kernels, CUDA
#   2 selftests  records / driver / trainer, against fakes and a toy model
#   3 probe      Qwen3.5-2B + LoRA fwd/bwd at 12k and 24k tokens, peak memory
#   4 server     serve-train.sh healthy
#   5 lora       a random adapter changes served logprobs; unload; reload from a new dir
#   5b swap      8 swaps via trainer.swap_adapter: served weights track a fresh load (prefix cache)
#   6 parity     zero-adapter trainer logprobs vs vLLM's on recorded base episodes
#   7 prefix     prefix-consistency gate over the profiling records
#   8 e2e        a 2-step run on 4 dev tasks, SIGKILLed mid-rollout of step 1, verified, resumed:
#                checkpoint durability, resume, and the logprob gap under a trained adapter
set -uo pipefail
cd "$(dirname "$0")/.."
TR=baseline/.venv-train/bin/python
TP=baseline/.venv-tau2/bin/python
declare -a RES
t() { local name=$1; shift; local t0=$SECONDS
      echo; echo "=== $name  $(date +%T)"
      if "$@"; then RES+=("PASS  $name  $((SECONDS - t0))s"); else RES+=("FAIL  $name  $((SECONDS - t0))s"); fi; }

t env      $TR baseline/check_train_env.py
t records  $TP records.py selftest
t driver   $TP driver.py selftest
t trainer  $TR trainer.py selftest
t probe12k $TR trainer.py probe --seq-len 12288
t probe24k $TR trainer.py probe --seq-len 24576
t server   baseline/session.sh

lora_smoke() {
  local d=$PWD/runs/pretest/rand-$(date +%s)   # absolute: the server runs from baseline/
  $TR trainer.py init-adapter "$d" --random || return 1
  $TR - "$d" <<'PY'
import json, sys, urllib.request
root = "http://localhost:8000"
def post(path, body):
    r = urllib.request.Request(root + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=300) as f:
        return f.read().decode()
# Score a FIXED text (echo=True returns every prompt token's logprob) so all calls compare the
# same tokens. Comparing generated tokens is invalid: a near-tie argmax flip changes the tokens.
TEXT = ("Hi, I'd like to return the water bottle and the desk lamp from order #W1234567. "
        "The lamp arrived with a cracked base and the bottle leaks. Can you refund it to my "
        "original credit card? My email is jane.doe@example.com and my zip code is 92101.")
def lps(model):
    out = json.loads(post("/v1/completions", {"model": model, "prompt": TEXT, "echo": True,
                                              "max_tokens": 1, "temperature": 0, "logprobs": 1}))
    return out["choices"][0]["logprobs"]["token_logprobs"][1:len(out["choices"][0]["logprobs"]["token_logprobs"]) - 1]
def mad(a, b):
    return sum(abs(x - y) for x, y in zip(a, b)) / len(a)
d = sys.argv[1]
try: post("/v1/unload_lora_adapter", {"lora_name": "policy"})
except Exception: pass
base = lps("qwen3.5-2B")
post("/v1/load_lora_adapter", {"lora_name": "policy", "lora_path": d + "/serve"})
a = lps("policy")
a2 = lps("policy")
post("/v1/unload_lora_adapter", {"lora_name": "policy"})
post("/v1/load_lora_adapter", {"lora_name": "policy", "lora_path": d + "/serve"})
b = lps("policy")
post("/v1/unload_lora_adapter", {"lora_name": "policy"})
r = {"tokens": len(base), "effect": mad(base, a), "repeat_noise": mad(a, a2), "reload": mad(a, b),
     "reload_vs_base": mad(b, base)}
print(json.dumps(r))
# Measured Sept 24: base serving is bit-exact across requests, but vLLM's LoRA path is not. A
# large random adapter moves logprobs ~0.045 mean |dlogp|/token, and repeats of the SAME request
# vary by 0.007-0.022, so no fixed ratio to that noise is a stable test. What must hold: the
# adapter is applied (base is bit-exact, so 0.02 cannot be noise), and a reload serves the
# adapter rather than base (closer to the first load than to base). lp_gap watches the noise.
sys.exit(0 if r["effect"] > 0.02 and r["reload"] < r["reload_vs_base"] else 1)
PY
}
t lora     lora_smoke
swap_fresh() {
  # The Sept 25 stale-adapter bug: under ONE reused served name, vLLM's prefix cache kept serving
  # KV/DeltaNet state from an older adapter, so rollouts lagged training by ~10 steps. Swap 8
  # clearly different adapters through trainer.swap_adapter (the loop's own code) and require
  # that what is served tracks a fresh load of the same weights, through the prefix cache.
  $TR - <<'PY'
import json, shutil, sys, urllib.request
from pathlib import Path
from safetensors.torch import load_file, save_file
sys.path.insert(0, ".")
from trainer import Config, swap_adapter
cfg = Config(); root = "http://localhost:8000"
src = sorted(Path("runs/pretest").glob("rand-*"))[-1]
work = Path("runs/pretest/swap"); shutil.rmtree(work, ignore_errors=True)
PREFIX = "You are a customer service agent for an online retail store. " + " ".join(
    f"Policy rule {i}: when a customer asks about item {1000 + i}, verify the order status, check the "
    f"payment method on file, and confirm before taking any action." for i in range(60))
SUF = [f" Customer message {j}: I want to return item {1000 + 7 * j} from my order. Agent:" for j in range(12)]
def post(path, body):
    r = urllib.request.Request(root + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return urllib.request.urlopen(r, timeout=300).read()
def lps(model):
    out = []
    for s in SUF:
        c = json.loads(post("/v1/completions", {"model": model, "prompt": PREFIX + s, "max_tokens": 4,
                                                "temperature": 0, "logprobs": 5}))["choices"][0]["logprobs"]
        out += sorted(c["top_logprobs"][0].values(), reverse=True)[:5]
    return out
mad = lambda a, b: sum(abs(x - y) for x, y in zip(a, b)) / len(a)
gaps = []
for k in range(1, 9):
    d = work / f"v{k}"; (d / "serve").mkdir(parents=True)
    for f in (src / "serve").iterdir():
        shutil.copy(f, d / "serve" / f.name)
    sd = load_file(str(src / "serve" / "adapter_model.safetensors"))
    save_file({n: (t * 0.5 * k if "lora_B" in n else t) for n, t in sd.items()}, str(d / "serve" / "adapter_model.safetensors"))
    name = swap_adapter(cfg, d.resolve(), f"swaptest{k}")
    served = lps(name)
    post("/v1/load_lora_adapter", {"lora_name": f"fresh{k}", "lora_path": str((d / "serve").resolve())})
    gaps.append(mad(served, lps(f"fresh{k}")))
    post("/v1/unload_lora_adapter", {"lora_name": f"fresh{k}"})
post("/v1/unload_lora_adapter", {"lora_name": name})
late = sum(gaps[4:]) / 4
print(json.dumps({"served_vs_fresh": [round(g, 4) for g in gaps], "mean_swaps_5_8": round(late, 4)}))
# measured Sept 25: reused name 0.065 and rising, unique names 0.032 and flat
sys.exit(0 if late < 0.045 else 1)
PY
}
t swap     swap_fresh
t parity   $TR trainer.py parity records/profile --limit 16
t prefix   bash -c "$TP records.py gate records/profile | head -11"

e2e() {
  local RUN=pretest-e2e-$(date +%s)
  $TR - "$RUN" <<'PY' || return 1
import json, sys
c = json.load(open("runs/p3-dev16b.config.json"))
c.update(tasks=c["tasks"][:4], steps=2, episodes_per_step=32, concurrency=32)
json.dump(c, open(f"runs/{sys.argv[1]}.config.json", "w"), indent=1)
PY
  $TR trainer.py run --run $RUN --config runs/$RUN.config.json &
  local PID=$!
  until [ -f runs/$RUN/manifests/b0001.json ]; do
    kill -0 $PID 2>/dev/null || { echo "trainer exited before step 1"; return 1; }; sleep 5
  done
  sleep 30
  echo "KILL -9 mid-rollout of step 1"
  pkill -9 -f "trainer.py run --run $RUN"; pkill -9 -f "runs/$RUN/manifests"; sleep 3
  RUN=$RUN bash baseline/verify_resume.sh || return 1
  $TR trainer.py run --run $RUN --config runs/$RUN.config.json --resume || return 1
  $TR - "$RUN" <<'PY'
import json, sys
rows = [json.loads(l) for l in open(f"runs/{sys.argv[1]}/metrics.jsonl") if l.strip()]
for r in rows:
    print({k: r.get(k) for k in ("step", "mean_reward", "episodes", "lp_gap_median", "lp_gap_p99",
                                 "tis_truncated_frac", "policy_version_rollout")})
ok = [r["step"] for r in rows] == [0, 1] and rows[1]["lp_gap_median"] < 0.01 and rows[1]["lp_gap_p99"] < 0.3
sys.exit(0 if ok else 1)
PY
}
t e2e      e2e

echo; echo "=== summary $(date +%T)"; printf '%s\n' "${RES[@]}"
echo "done"
