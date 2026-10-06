#!/usr/bin/env bash
# Deploy the per-Spark monitoring container to ONE Spark box.
#
#   ./deploy.sh root@192.168.1.107     # head Spark (also the box that runs the load harness)
#   ./deploy.sh root@192.168.1.108     # EdgeXpert worker
#
# ONE container per Spark. Each serves the same exporter on :9400, so the board reads the
# PAIR rather than only the box it happens to run on (see live_metrics.DEFAULT_BOXES) and
# Prometheus scrapes both. Idempotent: re-running rebuilds and recreates the container.
#
# The two Python modules are taken from the CHECKOUT ROOT (this script lives in monitor/),
# so the container always ships the same exporter the rest of the repo tests.
#
# NOTE (.107 only): the legacy host-side exporter `zgx-exporter.service` binds the same
# :9400 and must be stopped + DISABLED before this container can start, or the container
# cannot bind the port. That step is deliberately NOT automated -- it changes what runs on a
# production box, so the switch should be a conscious act. It is also the whole rollback:
#
#   ssh root@192.168.1.107 \
#     'systemctl disable --now zgx-exporter.service; \
#      docker rm -f spark-monitor; systemctl enable --now zgx-exporter.service'
set -euo pipefail

TARGET="${1:?usage: deploy.sh <user@host>}"
HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="$(cd "$HERE/.." && pwd)"
DEST=/root/spark-monitor

for f in "$SRC/zgx_exporter.py" "$SRC/corrected_metrics.py"; do
  [ -f "$f" ] || { echo "missing $f -- run this from inside the runboard-lcp checkout" >&2; exit 1; }
done

echo "deploy monitor -> $TARGET:$DEST"
ssh -o ConnectTimeout=10 "$TARGET" "mkdir -p $DEST"
scp -q "$SRC/zgx_exporter.py" "$SRC/corrected_metrics.py" \
       "$HERE/Dockerfile" "$HERE/docker-compose.monitor.yml" "$TARGET:$DEST/"

ssh -o ConnectTimeout=10 "$TARGET" \
  "cd $DEST && docker compose -f docker-compose.monitor.yml up -d --build"

echo "--- $TARGET ---"
ssh -o ConnectTimeout=10 "$TARGET" '
  docker ps --filter name=spark-monitor --format "  {{.Names}} {{.Status}}"
  printf "  /health  : "; curl -s --max-time 5 http://127.0.0.1:9400/health || echo FAIL; echo
  printf "  series   : "; curl -s --max-time 5 http://127.0.0.1:9400/metrics | grep -c "^zgx_" || echo FAIL
  printf "  temp     : "; curl -s --max-time 5 http://127.0.0.1:9400/metrics | grep -E "^zgx_gpu_temperature_celsius " || echo FAIL
'
