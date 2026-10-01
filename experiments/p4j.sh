#!/usr/bin/env bash
# p4-j pipeline on the VM (Sept 28): build v10, three branches from soup4, average, one valbig read,
# then re-score every read under tau3's rule. Idempotent: rerun after a preemption and each piece
# resumes or skips.
set -uo pipefail
cd ~
export API_BUDGET_USD=${API_BUDGET_USD:?set API_BUDGET_USD (the spend guard)}
T=baseline/.venv-tau2/bin/python; R=baseline/.venv-train/bin/python
[ -f tasks/profile-v10.json ] || $T taskgen/build_v10.py || exit 1
[ -f runs/p4-j3.config.json ] || $R experiments/p4j_config.py || exit 1
baseline/session.sh || exit 1
for i in 1 2 3; do
  ls runs/p4-j$i/adapters/004-* >/dev/null 2>&1 && continue
  RUN=p4-j$i baseline/train_run.sh || { echo "BRANCH $i FAILED"; exit 1; }
done
[ -f runs/soup/adapters/001-soupj/serve/adapter_model.safetensors ] || $R experiments/soup_j.py || exit 1
[ -f runs/valbig/soupj/_batch0401.summary.json ] || ls runs/valbig/soupj/_batch*.summary.json >/dev/null 2>&1 \
  || $R experiments/valbig_read_soupj.py || exit 1
$T experiments/rescore_val.py
echo P4J-DONE
