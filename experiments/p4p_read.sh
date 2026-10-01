#!/usr/bin/env bash
# gpt-5.2 proxy read of p4-p's final adapter. Idempotent.
set -uo pipefail
cd ~
export API_BUDGET_USD=${API_BUDGET_USD:?set API_BUDGET_USD (the spend guard)}
T=baseline/.venv-tau2/bin/python; R=baseline/.venv-train/bin/python
baseline/session.sh || exit 1
ls runs/proxy52/p4p/_batch*.summary.json >/dev/null 2>&1 || $R experiments/proxy52_p4p.py || { echo "READ FAILED"; exit 1; }
$T experiments/proxy52p_read.py
echo PX52P-DONE
