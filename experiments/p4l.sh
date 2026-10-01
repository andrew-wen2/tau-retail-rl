#!/usr/bin/env bash
# p4-l pipeline on the VM (Sept 29): build v12 -> customer pre-flight -> one branch from soup4
# (customer mix, lr 4e-5, repeat penalty, v12 pool). No val read.
# Idempotent.
set -uo pipefail
cd ~
export API_BUDGET_USD=${API_BUDGET_USD:?set API_BUDGET_USD (the spend guard)}
T=baseline/.venv-tau2/bin/python; R=baseline/.venv-train/bin/python
[ -f tasks/split_v12.json ] || $T taskgen/build_v12.py || { echo "BUILD FAILED"; exit 1; }
baseline/session.sh || exit 1
[ -f runs/p4-l.config.json ] || $R experiments/p4l_config.py || exit 1
if [ ! -f runs/custvet_p4l/VET-OK ]; then
  $R experiments/custvet_p4l.py || { echo "VET FAILED"; exit 1; }
  touch runs/custvet_p4l/VET-OK
fi
ls runs/p4-l/adapters/006-* >/dev/null 2>&1 || RUN=p4-l baseline/train_run.sh || { echo "BRANCH FAILED"; exit 1; }
echo P4L-DONE
