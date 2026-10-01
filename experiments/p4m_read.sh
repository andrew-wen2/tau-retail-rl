#!/usr/bin/env bash
# gpt-5.2 proxy read of soupm (soup4 + p4-m 003). Idempotent.
set -uo pipefail
cd ~
export API_BUDGET_USD=${API_BUDGET_USD:?set API_BUDGET_USD (the spend guard)}
T=baseline/.venv-tau2/bin/python; R=baseline/.venv-train/bin/python
baseline/session.sh || exit 1
[ -f runs/soup/adapters/003-soupm/serve/adapter_model.safetensors ] || $R experiments/soup_m.py || exit 1
ls runs/proxy52/soupm/_batch*.summary.json >/dev/null 2>&1 || $R experiments/proxy52_soupm.py || { echo "READ FAILED"; exit 1; }
$T experiments/proxy52m_read.py
echo PX52M-DONE
