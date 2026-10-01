#!/usr/bin/env bash
# VM-side bring-up, safe to re-run after any (re)start or Spot preemption:
#   1. the trainer venv, if check_train_env.py does not already pass
#   2. the training vLLM server (serve-train.sh), if :8000 is not already healthy
# Everything long-running is detached with setsid and stdin from /dev/null, so an ssh session
# that launches it returns immediately instead of hanging on the child's open descriptors.
#
# Usage (on the VM): baseline/session.sh           venv + server, wait for health
#                    NO_SERVER=1 baseline/session.sh
set -euo pipefail
cd "$(dirname "$0")"

if ! .venv-train/bin/python check_train_env.py >/dev/null 2>&1; then
  echo "session: building .venv-train (log: train_setup.log)"
  TRAIN_ONLY=1 ./setup.sh > train_setup.log 2>&1 || { tail -30 train_setup.log; exit 1; }
fi
.venv-train/bin/python check_train_env.py

[ "${NO_SERVER:-0}" = 1 ] && exit 0
if ! curl -sf localhost:8000/health >/dev/null; then
  echo "session: starting serve-train.sh (log: serve-train.log)"
  setsid nohup ./serve-train.sh 2B > serve-train.log 2>&1 < /dev/null &
fi
for _ in $(seq 120); do curl -sf localhost:8000/health >/dev/null && { echo "server healthy"; exit 0; }; sleep 5; done
echo "server did not come up in 10 min" >&2; tail -30 serve-train.log >&2; exit 1
