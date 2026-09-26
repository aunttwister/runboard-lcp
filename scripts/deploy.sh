#!/usr/bin/env bash
# Ship this checkout to the ZGX box and restart the read-only readers.
#
# The deployment is flat (/root/load/*.py, /root/load/*.html) while the checkout keeps pages
# under static/ -- that is the only layout difference. Nothing here touches the serving engine:
# engine switches go through serve.sh or the dispatcher, never through a deploy.
set -euo pipefail

HOST="${RUNBOARD_HOST:-root@192.168.1.107}"
LOAD="${RUNBOARD_LOAD:-/root/load}"
SRC="$(cd "$(dirname "$0")/.." && pwd)"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"

MODULES=(console_core.py registry.py dispatcher.py live_server.py history_collector.py
         zgx_exporter.py corrected_metrics.py live_metrics.py ui_chrome.py
         run_throughput.py)
PAGES=(index.html history.html console.html)

echo "deploy $SRC -> $HOST:$LOAD (backup $LOAD/backup-$STAMP)"
ssh -o ConnectTimeout=8 "$HOST" "mkdir -p '$LOAD/backup-$STAMP'"

for f in "${MODULES[@]}"; do
  scp -q -o ConnectTimeout=8 "$SRC/$f" "$HOST:$LOAD/.$f.new"
  ssh -o ConnectTimeout=8 "$HOST" \
    "cp -f '$LOAD/$f' '$LOAD/backup-$STAMP/$f' 2>/dev/null || true; mv -f '$LOAD/.$f.new' '$LOAD/$f'"
  echo "  sent $f"
done

for f in "${PAGES[@]}"; do
  scp -q -o ConnectTimeout=8 "$SRC/static/$f" "$HOST:$LOAD/.$f.new"
  ssh -o ConnectTimeout=8 "$HOST" \
    "cp -f '$LOAD/$f' '$LOAD/backup-$STAMP/$f' 2>/dev/null || true; mv -f '$LOAD/.$f.new' '$LOAD/$f'"
  echo "  sent $f"
done

# syntax gate on the far side BEFORE restarting anything
echo "compile gate:"
ssh -o ConnectTimeout=8 "$HOST" "cd '$LOAD' && for f in ${MODULES[*]}; do python3 -m py_compile \$f || exit 1; done && echo '  all modules compile'"

ssh -o ConnectTimeout=8 "$HOST" "systemctl restart load-live.service && sleep 2"
echo "load-live.service: $(ssh -o ConnectTimeout=8 "$HOST" 'systemctl is-active load-live.service')"

# reader smoke: the pages must still answer, and the console's JSON APIs must still parse
echo "endpoint smoke:"
for path in / /history /console /api/state /api/history /api/models /api/live; do
  code=$(ssh -o ConnectTimeout=8 "$HOST" "curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:18400$path")
  echo "  $path -> $code"
done

echo "done. rollback: ssh $HOST 'cp $LOAD/backup-$STAMP/* $LOAD/' && ssh $HOST 'systemctl restart load-live.service'"
