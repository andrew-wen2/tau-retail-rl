#!/usr/bin/env bash
# Check that a run can be resumed after a preemption or kill: the checkpoint and the adapter it
# names load, every metrics row parses, and no recent text file has a NUL-filled tail (an append
# or rename whose data never reached disk). Exits non-zero on any damage. From the repo root:
#   RUN=p3-dev16b baseline/verify_resume.sh
cd "$(dirname "$0")/.." || exit 1
R=runs/${RUN:?set RUN}
bad=0
for f in $(find . \( -path ./baseline/.venv\* -o -path ./.cache -o -path ./.triton \) -prune -o -type f -mmin -120 \
            \( -name '*.json' -o -name '*.jsonl' -o -name '*.py' -o -name '*.sh' \) -print); do
  grep -qaP '\x00' "$f" && { echo "NUL: $f"; bad=1; }
done
baseline/.venv-train/bin/python - "$R" <<'PY' || bad=1
import json, sys, torch
from pathlib import Path
from safetensors.torch import load_file
r = Path(sys.argv[1])
m = r / "metrics.jsonl"
rows = [json.loads(l) for l in m.read_text().splitlines() if l.strip()] if m.exists() else []
print("metrics steps", [x["step"] for x in rows])
ck = r / "ckpt" / "state.pt"
if ck.exists():
    st = torch.load(ck, weights_only=False)
    a = r / "adapters" / st["policy_version"]
    load_file(a / "adapter_model.safetensors"); load_file(a / "serve" / "adapter_model.safetensors")
    print("ckpt step", st["step"], "last_batch", st["last_batch_id"], "policy", st["policy_version"], "adapter ok")
    if rows and rows[-1]["step"] != st["step"] - 1:
        print("WARN metrics last step", rows[-1]["step"], "vs ckpt step", st["step"])
else:
    print("no checkpoint; run restarts from step 0")
PY
[ $bad = 0 ] && echo VERIFY-OK || echo VERIFY-FAILED
exit $bad
