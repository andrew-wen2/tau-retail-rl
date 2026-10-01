#!/usr/bin/env bash
# Laptop-side helper for the Spot A100. Run from anywhere on the Mac.
#
#   ./vm.sh up        start the VM, retrying until Spot capacity comes back
#   ./vm.sh sync      push the repo (minus .git) into the VM's home, which mirrors the repo root
#   ./vm.sh ssh [cmd] ssh in, or run one command
#   ./vm.sh pull PATH copy PATH (relative to the VM home) back into the repo
#   ./vm.sh down      stop the VM when idle
set -euo pipefail
VM=${VM:-rl-a100-a}
ZONE=${ZONE:-us-central1-a}
USER_AT=awen2815@$VM
REPO="$(cd "$(dirname "$0")/.." && pwd)"

status() { gcloud compute instances describe "$VM" --zone "$ZONE" --format='value(status)'; }

case "${1:-}" in
  up)
    # Spot capacity in us-central1-a disappears for hours at a time. A failed
    # start is ZONE_RESOURCE_POOL_EXHAUSTED, not an error worth stopping on, so retry with
    # capped backoff until it runs or MAX_WAIT seconds pass.
    MAX_WAIT=${MAX_WAIT:-21600}; delay=60; waited=0
    until [ "$(status)" = RUNNING ]; do
      if gcloud compute instances start "$VM" --zone "$ZONE" >/dev/null 2>/tmp/vm-start.err; then
        break
      fi
      echo "$(date +%T) start failed: $(tail -1 /tmp/vm-start.err | cut -c1-120)" >&2
      [ "$waited" -ge "$MAX_WAIT" ] && { echo "gave up after ${waited}s" >&2; exit 1; }
      sleep "$delay"; waited=$((waited + delay)); delay=$((delay < 600 ? delay * 2 : 600))
    done
    # sshd lags the RUNNING state by a few seconds.
    until gcloud compute ssh "$USER_AT" --zone "$ZONE" --command true >/dev/null 2>&1; do sleep 5; done
    echo "up: $VM" ;;
  sync)
    COPYFILE_DISABLE=1 tar -C "$REPO" --exclude .git --exclude __pycache__ --exclude .DS_Store -czf - . |
      gcloud compute ssh "$USER_AT" --zone "$ZONE" --command 'tar -xzf - -C ~' ;;
  ssh)
    shift; gcloud compute ssh "$USER_AT" --zone "$ZONE" ${1:+--command "$*"} ;;
  pull)
    gcloud compute scp --recurse --zone "$ZONE" "$USER_AT:~/${2:?path}" "$REPO/$(dirname "$2")/" ;;
  down)
    gcloud compute instances stop "$VM" --zone "$ZONE" --discard-local-ssd=true ;;
  *)
    sed -n 2,9p "$0"; exit 2 ;;
esac
