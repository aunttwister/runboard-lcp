#!/bin/bash
# serve.sh — move :18300 between the engines that can actually hold it.
#
#   serve.sh status      what is serving now, and which build is loaded
#   serve.sh glm53       TensorFold v0.6.0 + GLM-5.3-Flash EXL3 4bpw (2x Spark TP=2) [current default]
#   serve.sh cruz        the Cruz fork (523ecd3) + turboderp 3.05bpw
#   serve.sh exl3        stock exllamav3 + r0b0tlab 2.50bpw
#   serve.sh vllm        the vLLM prod container (NVFP4 + abliterated)
#   serve.sh vllm-cruz   vLLM + vllm-exl3 fork + turboderp 3.05bpw
#   serve.sh tensorfold  TensorFold v0.3.6.3 + Vontra MLX 4-bit MTP (container)
#
# Why one command: the engines cannot coexist (vLLM ~94 GB, 3.05bpw ~85 GB, 2.50bpw ~61 GB
# against 121 GB unified), so ":18300" is served by exactly one of them. Every switch
# health-gates and, on failure, restores the engine that was serving before.
set -u
PORT=18300
CRUZ=exl3-cruz-fork.service
VLLM_EXL3=vllm-exl3-cruz.service
STOCK=exl3-2.5bpw.service
TF=tensorfold.service
PROD=vllm-fn-tp1
GLM=glm53-flash-tf
LOG=/root/serve.log
RESTORE=""

step(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }
health(){ curl -sf --max-time 6 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; }
current(){
  if systemctl is-active --quiet "$TF"; then echo tensorfold
  elif systemctl is-active --quiet "$VLLM_EXL3"; then echo vllm-cruz
  elif systemctl is-active --quiet "$CRUZ"; then echo cruz
  elif systemctl is-active --quiet "$STOCK"; then echo exl3
  elif [ "$(docker inspect -f '{{.State.Status}}' "$PROD" 2>/dev/null)" = running ]; then echo vllm
  elif [ "$(docker inspect -f '{{.State.Status}}' "$GLM" 2>/dev/null)" = running ]; then echo glm53
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
  echo "  units        : $VLLM_EXL3=$(systemctl is-active $VLLM_EXL3 2>&1) enabled=$(systemctl is-enabled $VLLM_EXL3 2>&1)"
  echo "                 $CRUZ=$(systemctl is-active $CRUZ 2>&1) enabled=$(systemctl is-enabled $CRUZ 2>&1)"
  echo "                 $STOCK=$(systemctl is-active $STOCK 2>&1) enabled=$(systemctl is-enabled $STOCK 2>&1)"
  echo "                 $TF=$(systemctl is-active $TF 2>&1) enabled=$(systemctl is-enabled $TF 2>&1)"
  echo "  container    : $PROD=$(docker inspect -f '{{.State.Status}}' $PROD 2>/dev/null || echo absent)"
  echo "                 glm53-flash-tf=$(docker inspect -f '{{.State.Status}}' glm53-flash-tf 2>/dev/null || echo absent)"
  echo "                 qwen38-flash-next-tf=$(docker inspect -f '{{.State.Status}}' qwen38-flash-next-tf 2>/dev/null || echo absent)"
  echo "  model id     : $(curl -s --max-time 6 http://127.0.0.1:$PORT/v1/models 2>/dev/null \
      | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null || echo '(no answer)')"
  free -g | head -2 | sed 's/^/  /'
}

stop_exl3(){ systemctl stop "$CRUZ" "$STOCK" >/dev/null 2>&1 || true; pkill -f 'serve_openai[.]py' 2>/dev/null || true; sleep 8; }
stop_vllm(){ docker stop -t 60 "$PROD" >/dev/null 2>&1 || true; sleep 8; }
# The TensorFold unit wraps the recipe's own start.sh; stopping the unit runs stop.sh
# through start.sh's INT/TERM trap, which removes the container and frees its GPU memory.
stop_tensorfold(){ systemctl stop "$TF" >/dev/null 2>&1 || true; sleep 12; }
# The GLM-5.3-Flash kit is a plain container (restart policy "no"): docker stop is the whole
# lifecycle, and it maps ~160 GB of weights across two ranks, so give the shutdown a longer
# grace than the prod container -- then wait for the port itself, same rule as below.
stop_glm(){
  docker stop -t 120 "$GLM" >/dev/null 2>&1 || true
  local i
  for i in $(seq 1 30); do
    ss -ltn 2>/dev/null | grep -q ":$PORT " || { sleep 2; return 0; }
    sleep 2
  done
  step "!! :$PORT is still held after stopping $GLM"
  return 0
}
# vllm-exl3-cruz.service is the default occupant of :18300, and NOTHING here used to stop
# it: stop_vllm only stops the PROD *container*, and stop_exl3 only the fork/stock
# exllamav3 units. Every switch away from the baseline therefore left the incumbent
# listening, and the incoming engine aborted with "port 18300 is already in use" (or never
# answered at all) while serve.sh still exited 0 -- the whole reason a TensorFold switch
# could "succeed" without starting anything.
#
# Wait for the port itself rather than guessing a sleep: releasing ~85 GB of weights takes
# as long as it takes, and a free port is the thing the next engine actually needs.
stop_vllm_cruz(){
  systemctl stop "$VLLM_EXL3" >/dev/null 2>&1 || true
  local i
  for i in $(seq 1 30); do
    ss -ltn 2>/dev/null | grep -q ":$PORT " || { sleep 2; return 0; }
    sleep 2
  done
  step "!! :$PORT is still held after stopping $VLLM_EXL3"
  return 0
}

wait_health(){ # $1 label, $2 tries(10s)
  local i
  for i in $(seq 1 "${2:-90}"); do
    sleep 10
    health && { step "$1 HEALTHY after $((i*10))s"; return 0; }
  done
  return 1
}

# Same as wait_health, but also gives up as soon as the unit backing the engine is no
# longer active. Type=simple reports the unit as started the moment ExecStart is
# spawned, so a start.sh that dies immediately (missing image, port already in use,
# checkpoint not downloaded) would otherwise burn the whole 30-minute budget before
# failing. `systemctl is-active` after the fact is what turns that into a fast, named
# failure instead of a silent half-hour stall.
wait_health_unit(){ # $1 label, $2 tries(10s), $3 unit
  local i
  for i in $(seq 1 "${2:-90}"); do
    sleep 10
    health && { step "$1 HEALTHY after $((i*10))s"; return 0; }
    if ! systemctl is-active --quiet "$3"; then
      step "$1: $3 is no longer active after $((i*10))s (start.sh exited) — giving up"
      return 1
    fi
  done
  return 1
}

restore_on_failure(){
  [ -n "$RESTORE" ] || return 0
  step "!! target failed to come up — restoring '$RESTORE' (the previous state)"
  case "$RESTORE" in
    vllm-cruz) systemctl reset-failed "$VLLM_EXL3" >/dev/null 2>&1; systemctl start "$VLLM_EXL3" ;;
    cruz) systemctl reset-failed "$CRUZ" >/dev/null 2>&1; systemctl start "$CRUZ" ;;
    exl3) systemctl reset-failed "$STOCK" >/dev/null 2>&1; systemctl start "$STOCK" ;;
    vllm) docker start "$PROD" >/dev/null 2>&1 || (cd /root/miaai-qwen3.8-single-dgx && nohup ./launch-seqs16.sh >/root/prod-restore.log 2>&1 &) ;;
    tensorfold) systemctl reset-failed "$TF" >/dev/null 2>&1; systemctl start "$TF" ;;
    glm53) docker start "$GLM" >/dev/null 2>&1 || true ;;
  esac
  wait_health "restored $RESTORE" 90 || step "!! the previous engine did not come back either — :$PORT is empty"
}

case "${1:-status}" in
  status|"") show_status ;;
  vllm-cruz)
    RESTORE=$(current); [ "$RESTORE" = vllm-cruz ] && { step "already serving vLLM + vllm-exl3"; show_status; exit 0; }
    step "=== switch to vLLM + vllm-exl3 + 3.05bpw (was: $RESTORE) ==="
    stop_vllm; stop_exl3; stop_tensorfold; stop_glm
    systemctl reset-failed "$VLLM_EXL3" >/dev/null 2>&1 || true
    systemctl start "$VLLM_EXL3" || { step "ABORT: start failed"; restore_on_failure; exit 4; }
    # ~85 GB of exl3 pack plus per-expert post-processing: about 11 minutes,
    # so this budget is deliberately larger than the exllamav3 arms (90).
    wait_health "vLLM + vllm-exl3 3.05bpw" 110 || { restore_on_failure; exit 5; }
    step "rollback = serve.sh $RESTORE"
    show_status ;;
  cruz)
    RESTORE=$(current); [ "$RESTORE" = cruz ] && { step "already serving cruz"; show_status; exit 0; }
    step "=== switch to CRUZ FORK + 3.05bpw (was: $RESTORE) ==="
    stop_vllm; stop_exl3; stop_tensorfold; stop_vllm_cruz; stop_glm
    systemctl reset-failed "$CRUZ" >/dev/null 2>&1 || true
    systemctl start "$CRUZ" || { step "ABORT: start failed"; restore_on_failure; exit 4; }
    wait_health "cruz fork" 90 || { restore_on_failure; exit 5; }
    step "rollback = serve.sh $RESTORE"
    show_status ;;
  exl3)
    RESTORE=$(current); [ "$RESTORE" = exl3 ] && { step "already serving exl3 2.50bpw"; show_status; exit 0; }
    step "=== switch to stock exllamav3 + 2.50bpw (was: $RESTORE) ==="
    stop_vllm; stop_exl3; stop_tensorfold; stop_vllm_cruz; stop_glm
    systemctl reset-failed "$STOCK" >/dev/null 2>&1 || true
    systemctl start "$STOCK" || { step "ABORT: start failed"; restore_on_failure; exit 4; }
    wait_health "exl3 2.50bpw" 90 || { restore_on_failure; exit 5; }
    step "rollback = serve.sh $RESTORE"
    show_status ;;
  vllm)
    RESTORE=$(current); [ "$RESTORE" = vllm ] && { step "already serving vLLM"; show_status; exit 0; }
    step "=== switch to the vLLM prod container (was: $RESTORE) ==="
    stop_exl3; stop_tensorfold; stop_vllm_cruz; stop_glm
    docker start "$PROD" >/dev/null 2>&1 || { step "ABORT: docker start failed"; restore_on_failure; exit 4; }
    wait_health "vLLM prod" 120 || { restore_on_failure; exit 5; }
    step "rollback = serve.sh $RESTORE"
    show_status ;;
  tensorfold)
    RESTORE=$(current); [ "$RESTORE" = tensorfold ] && { step "already serving TensorFold"; show_status; exit 0; }
    step "=== switch to TensorFold v0.3.6.3 + MLX 4-bit (was: $RESTORE) ==="
    stop_vllm; stop_exl3; stop_tensorfold; stop_vllm_cruz; stop_glm
    systemctl reset-failed "$TF" >/dev/null 2>&1 || true
    systemctl start "$TF" || { step "ABORT: start failed"; restore_on_failure; exit 4; }
    # A warm start loads ~75 GiB of weights in ~2.5 min; the FIRST start after an image
    # change also compiles the CUDA kernels for GB10, which is why this budget (30 min)
    # is far larger than the exllamav3 arms (90) or vLLM (110) need. The unit-aware
    # variant fails immediately if start.sh itself exits (e.g. port taken, image missing).
    wait_health_unit "TensorFold" 180 "$TF" || { restore_on_failure; exit 5; }
    step "rollback = serve.sh $RESTORE"
    show_status ;;
  glm53)
    RESTORE=$(current); [ "$RESTORE" = glm53 ] && { step "already serving the GLM-5.3-Flash TensorFold kit"; show_status; exit 0; }
    step "=== switch to TensorFold v0.6.0 + GLM-5.3-Flash EXL3 4bpw, 2x Spark TP=2 (was: $RESTORE) ==="
    stop_vllm; stop_exl3; stop_tensorfold; stop_vllm_cruz
    docker start "$GLM" >/dev/null 2>&1 || { step "ABORT: docker start failed (container removed? recreate it)"; restore_on_failure; exit 4; }
    # Two ranks must each map ~80 GB before the port answers; a warm start is minutes, a
    # cold one (fresh page cache) is not. 30 minutes, like the other TensorFold arm.
    wait_health "TensorFold GLM-5.3-Flash kit" 180 || { restore_on_failure; exit 5; }
    step "rollback = serve.sh $RESTORE"
    show_status ;;
  *) echo "usage: serve.sh [status|glm53|cruz|vllm-cruz|exl3|vllm|tensorfold]"; exit 2 ;;
esac
