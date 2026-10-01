#!/usr/bin/env bash
# p4-o pipeline on the VM (Sept 29): p4-m at lr 3e-5, one branch from soup4, 6 x 192. Idempotent.
set -uo pipefail
cd ~
export API_BUDGET_USD=${API_BUDGET_USD:?set API_BUDGET_USD (the spend guard)}
R=baseline/.venv-train/bin/python
[ -f runs/p4-m.config.json ] || { echo "p4-m config missing; FAILED"; exit 1; }
baseline/session.sh || exit 1
[ -f runs/p4-o.config.json ] || $R experiments/p4o_config.py || exit 1
ls runs/p4-o/adapters/006-* >/dev/null 2>&1 || RUN=p4-o baseline/train_run.sh || { echo "BRANCH FAILED"; exit 1; }
echo P4O-DONE
