#!/usr/bin/env bash
# p4-k pipeline on the VM (Sept 29): light re-profile -> one graded-credit branch from soup4 ->
# average with p4-j1..3 -> valbig reads of the branch and the average -> re-score. Idempotent.
set -uo pipefail
cd ~
export API_BUDGET_USD=${API_BUDGET_USD:?set API_BUDGET_USD (the spend guard)}
T=baseline/.venv-tau2/bin/python; R=baseline/.venv-train/bin/python
[ -f tasks/to_profile_v11.json ] || $T taskgen/build_v11.py || exit 1
baseline/session.sh || exit 1
if [ ! -f tasks/profile-v11.json ]; then
  rm -rf runs/profile-v11
  PROFILE_TAG=v11 PROFILE_MERGE=add $R taskgen/profile_v6.py $HOME/runs/soup/adapters/000-soup4 4 || { echo "PROFILE FAILED"; exit 1; }
  echo "PROFILE-DONE"
fi
[ -f runs/p4-k.config.json ] || $R experiments/p4k_config.py || exit 1
ls runs/p4-k/adapters/006-* >/dev/null 2>&1 || RUN=p4-k baseline/train_run.sh || { echo "BRANCH FAILED"; exit 1; }
[ -f runs/soup/adapters/002-soupk/serve/adapter_model.safetensors ] || $R experiments/soup_k.py || exit 1
ls runs/valbig/p4k/_batch*.summary.json >/dev/null 2>&1 || $R experiments/valbig_read_p4k.py || { echo "READ FAILED"; exit 1; }
ls runs/valbig/soupk/_batch*.summary.json >/dev/null 2>&1 || $R experiments/valbig_read_soupk.py || { echo "READ FAILED"; exit 1; }
$T experiments/rescore_val.py
echo P4K-DONE
