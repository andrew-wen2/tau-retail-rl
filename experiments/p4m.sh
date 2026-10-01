#!/usr/bin/env bash
# p4-m pipeline on the VM (Sept 29): OpenAI-customer vet on bundle tasks -> one branch from soup4
# (DeepSeek T=1 + the chosen OpenAI customer, lr 4e-5, repeat penalty, v12 pool). Idempotent.
set -uo pipefail
cd ~
export API_BUDGET_USD=${API_BUDGET_USD:?set API_BUDGET_USD (the spend guard)}
R=baseline/.venv-train/bin/python
[ -f tasks/split_v12.json ] && [ -f runs/p4-l.config.json ] || { echo "v12 / p4-l config missing; FAILED"; exit 1; }
baseline/session.sh || exit 1
[ -f runs/custvet_p4m/choice.json ] || $R experiments/custvet_p4m.py || { echo "VET FAILED"; exit 1; }
[ -f runs/p4-m.config.json ] || $R experiments/p4m_config.py || exit 1
ls runs/p4-m/adapters/006-* >/dev/null 2>&1 || RUN=p4-m baseline/train_run.sh || { echo "BRANCH FAILED"; exit 1; }
echo P4M-DONE
