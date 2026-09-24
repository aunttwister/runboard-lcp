#!/bin/bash
# serve.sh — move :18300 between the engines that can actually hold it.
#
#   serve.sh status      what is serving now, and which build is loaded
#   serve.sh cruz        the Cruz fork (523ecd3) + turboderp 3.05bpw   [current default]
#   serve.sh exl3        stock exllamav3 + r0b0tlab 2.50bpw
#   serve.sh vllm        the vLLM prod container (NVFP4 + abliterated)
#
# Why one command: the engines cannot coexist (vLLM ~94 GB, 3.05bpw ~85 GB, 2.50bpw ~61 GB
# against 121 GB unified), so ":18300" is served by exactly one of them. Every switch
# health-gates and, on failure, restores the engine that was serving before.
set -u
PORT=18300
CRUZ=exl3-cruz-fork.service
STOCK=exl3-2.5bpw.service
PROD=vllm-fn-tp1
LOG=/root/serve.log
RESTORE=""

step(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }
health(){ curl -sf --max-time 6 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; }
current(){
  if systemctl is-active --quiet "$CRUZ"; then echo cruz
  elif systemctl is-active --quiet "$STOCK"; then echo exl3
  elif [ "$(docker inspect -f '{{.State.Status}}' "$PROD" 2>/dev/null)" = running ]; then echo vllm
  else echo none; fi
}

show_status(){
  local cur; cur=$(current)
  echo "serving :$PORT  ->  $cur"
  local pid; pid=$(ss -ltnp 2>/dev/null | grep ":$PORT " | grep -o 'pid=[0-9]*' | head -1 | cut -d= -f2)
  echo "  listener pid : ${pid:-none}"
  if [ -n "${pid:-}" ]; then
    local ext
    ext=$(grep -m1 -o '/root/exl3-engine/[^ ]*exllamav3_ext[^ ]*\.so' /proc/$pid/maps 2>/dev/null)
    case "$ext" in
      *cruz-exllamav3*) echo "  engine build : CRUZ FORK (523ecd3)" ;;
      *r0b0tlab*)       echo "  engine build : stock (r0b0tlab gb10)" ;;
      "")               echo "  engine build : not an exllamav3 process (vLLM?)" ;;
      *)                echo "  engine build : $ext" ;;
    esac
  fi
  echo "  units        : $CRUZ=$(systemctl is-active $CRUZ 2>&1) enabled=$(systemctl is-enabled $CRUZ 2>&1)"
  echo "                 $STOCK=$(systemctl is-active $STOCK 2>&1) enabled=$(systemctl is-enabled $STOCK 2>&1)"
  echo "  container    : $PROD=$(docker inspect -f '{{.State.Status}}' $PROD 2>/dev/null || echo absent)"
  echo "  model id     : $(curl -s --max-time 6 http://127.0.0.1:$PORT/v1/models 2>/dev/null \
      | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null || echo '(no answer)')"
  free -g | head -2 | sed 's/^/  /'
}

stop_exl3(){ systemctl stop "$CRUZ" "$STOCK" >/dev/null 2>&1 || true; pkill -f 'serve_openai[.]py' 2>/dev/null || true; sleep 8; }
stop_vllm(){ docker stop -t 60 "$PROD" >/dev/null 2>&1 || true; sleep 8; }

wait_health(){ # $1 label, $2 tries(10s)
  local i
  for i in $(seq 1 "${2:-90}"); do
    sleep 10
    health && { step "$1 HEALTHY after $((i*10))s"; return 0; }
  done
  return 1
}

restore_on_failure(){
  [ -n "$RESTORE" ] || return 0
  step "!! target failed to come up — restoring '$RESTORE' (the previous state)"
  case "$RESTORE" in
    cruz) systemctl reset-failed "$CRUZ" >/dev/null 2>&1; systemctl start "$CRUZ" ;;
    exl3) systemctl reset-failed "$STOCK" >/dev/null 2>&1; systemctl start "$STOCK" ;;
    vllm) docker start "$PROD" >/dev/null 2>&1 || (cd /root/miaai-qwen3.8-single-dgx && nohup ./launch-seqs16.sh >/root/prod-restore.log 2>&1 &) ;;
  esac
  wait_health "restored $RESTORE" 90 || step "!! the previous engine did not come back either — :$PORT is empty"
}

case "${1:-status}" in
  status|"") show_status ;;
  cruz)
    RESTORE=$(current); [ "$RESTORE" = cruz ] && { step "already serving cruz"; show_status; exit 0; }
    step "=== switch to CRUZ FORK + 3.05bpw (was: $RESTORE) ==="
    stop_vllm; stop_exl3
    systemctl reset-failed "$CRUZ" >/dev/null 2>&1 || true
    systemctl start "$CRUZ" || { step "ABORT: start failed"; restore_on_failure; exit 4; }
    wait_health "cruz fork" 90 || { restore_on_failure; exit 5; }
    step "rollback = serve.sh $RESTORE"
    show_status ;;
  exl3)
    RESTORE=$(current); [ "$RESTORE" = exl3 ] && { step "already serving exl3 2.50bpw"; show_status; exit 0; }
    step "=== switch to stock exllamav3 + 2.50bpw (was: $RESTORE) ==="
    stop_vllm; stop_exl3
    systemctl reset-failed "$STOCK" >/dev/null 2>&1 || true
    systemctl start "$STOCK" || { step "ABORT: start failed"; restore_on_failure; exit 4; }
    wait_health "exl3 2.50bpw" 90 || { restore_on_failure; exit 5; }
    step "rollback = serve.sh $RESTORE"
    show_status ;;
  vllm)
    RESTORE=$(current); [ "$RESTORE" = vllm ] && { step "already serving vLLM"; show_status; exit 0; }
    step "=== switch to the vLLM prod container (was: $RESTORE) ==="
    stop_exl3
    docker start "$PROD" >/dev/null 2>&1 || { step "ABORT: docker start failed"; restore_on_failure; exit 4; }
    wait_health "vLLM prod" 120 || { restore_on_failure; exit 5; }
    step "rollback = serve.sh $RESTORE"
    show_status ;;
  *) echo "usage: serve.sh [status|cruz|exl3|vllm]"; exit 2 ;;
esac
