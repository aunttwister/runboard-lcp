#!/bin/bash
# Capture a torch profile of the decode path on the GB10 serving box, then put
# the production unit back exactly as it was.
#
# Why a restart: vLLM exposes /start_profile only when --profiler-config was
# given at launch, and the running unit does not set it. The box is idle and
# there is no room for a second instance (84 GiB of weights + 11 GiB KV), so the
# only way to profile is to swap the process. Everything is restored in `finally`
# even if the profiled launch fails.
#
# Usage: bash profile_engine.sh [profile_dir]
set -uo pipefail

PROF_DIR="${1:-/root/prof_ext}"
UNIT="vllm-exl3-cruz.service"
LOG=/root/vllm-prof.log
URL=http://127.0.0.1:18300

wait_healthy() {
  local n=0
  while (( n < 120 )); do
    if curl -s --max-time 3 "$URL/health" >/dev/null 2>&1; then
      echo "  healthy after ~$((n * 5))s"
      return 0
    fi
    sleep 5; n=$((n + 1))
  done
  echo "  NOT healthy after 600s"; return 1
}

restore() {
  echo "=== restoring the production unit ==="
  pkill -f "profiler-config" 2>/dev/null
  pkill -f "vllm serve /root/models/Qwen3.8-Flash-Next-exl3-3.05bpw" 2>/dev/null
  sleep 3
  systemctl start "$UNIT"
  wait_healthy || echo "  WARNING: unit did not come back healthy -- check journalctl -u $UNIT"
  systemctl is-active "$UNIT"
}
trap restore EXIT

echo "=== stopping the production unit ==="
rm -rf "$PROF_DIR"; mkdir -p "$PROF_DIR"
systemctl stop "$UNIT"
sleep 5
pkill -f "vllm serve /root/models/Qwen3.8-Flash-Next-exl3-3.05bpw" 2>/dev/null || true
sleep 3

echo "=== launching the same server, profiler enabled ==="
export MODEL_DIR=/root/models/Qwen3.8-Flash-Next-exl3-3.05bpw
export HOST=0.0.0.0 PORT=18300 SERVED_NAME=qwen3.8-flash-next
export MAX_MODEL_LEN=262144 GPU_MEM_UTIL=0.80 MAX_NUM_SEQS=4
export NGRAM_TABLE=resident MAMBA_SSM_DTYPE=bfloat16
export SPEC_CONFIG='{"method":"mtp","num_speculative_tokens":3}'
export VLLM_EXL3_NGRAM_KERNEL=ext
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_EXTENSIONS_DIR=/root/venvs/vllm-exl3/torch_extensions
export TORCH_CUDA_ARCH_LIST=12.1
export PROFILER_DIR="$PROF_DIR"
export PATH="/root/venvs/vllm-exl3/bin:/usr/local/cuda-13.0/bin:$PATH"
nohup /bin/bash /root/exl3-engine/cruz-recipe/scripts/serve_one_spark_qwen.sh \
  >"$LOG" 2>&1 &
echo "  launch pid $!"
wait_healthy || { echo "profiled launch never became healthy"; exit 1; }

echo "=== profiling a decode burst ==="
curl -s -X POST "$URL/start_profile" -o /dev/null -w "  start_profile http=%{http_code}\n"
cd /root && timeout 600 python3 -u bench_decode.py --reps 2 --label profiled >/root/bench-profiled.txt 2>&1
tail -6 /root/bench-profiled.txt
curl -s -X POST "$URL/stop_profile" -o /dev/null -w "  stop_profile http=%{http_code}\n"
sleep 8

echo "=== trace files written ==="
find "$PROF_DIR" -type f | head -10
du -sh "$PROF_DIR" 2>/dev/null
echo "=== done; restore follows ==="
