#!/usr/bin/env bash
# Plan A — Traffic-only sweep tailored for an RTX 4090 (24 GB).
#
# Why this file exists: Traffic is the single biggest item in the main 6000-Ada
# queue (~800 GPU-hours). Running it on a second machine in parallel cuts the
# main wall-clock from ~58 days down to ~33 days. This script is meant to be
# scp'd to the 4090 machine together with the SimMTM_Forecasting/ directory and
# /workspace/data/signal/forecasting/traffic/ data.
#
# Usage on the 4090 box:
#   /usr/bin/python3 scripts/experiments/multi_dataset_sweep.py \
#       --datasets Traffic \
#       --models SimMTM,PatchTST,iTransformer \
#       --losses mse,sdsc,hybrid,dtw,pcc,snr,zcr,dilate \
#       --seeds 2023 \
#       --gpu 0
#
# This thin wrapper just runs the same driver with --datasets=Traffic.
# Resumable via outputs/experiments/multi_sweep_v1/run_status.tsv.
#
# Memory note: Traffic with batch_size=4, d_model=128, 862 channels and --use_amp
# should fit in 24GB. If you hit OOM, drop batch_size to 2 (edit datasets_config.py).
set -euo pipefail

ROOT_DEFAULT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ROOT="${SIMMTM_ROOT:-$ROOT_DEFAULT}"
cd "$ROOT"

# Optional: nohup so disconnect doesn't kill the run
LOG="/tmp/traffic_only_4090.log"

echo "Launching Traffic-only sweep from $ROOT"
echo "Output log: $LOG"
echo ""

nohup /usr/bin/python3 scripts/experiments/multi_dataset_sweep.py \
    --datasets Traffic \
    --models SimMTM,PatchTST,iTransformer \
    --losses mse,sdsc,hybrid,dtw,pcc,snr,zcr,dilate \
    --seeds 2023 \
    --gpu 0 \
    --out-dir "${ROOT}/outputs/experiments/multi_sweep_v1_4090" \
    > "$LOG" 2>&1 &

PID=$!
echo "$PID" > /tmp/traffic_only_4090.pid
disown $PID 2>/dev/null || true

echo "Sweep PID: $PID"
echo ""
echo "Monitor with:"
echo "  cat \$(cat /tmp/traffic_only_4090.pid 2>/dev/null) running? ps -p \$(cat /tmp/traffic_only_4090.pid)"
echo "  tail -f $LOG"
echo "  ls -lt outputs/experiments/multi_sweep_v1_4090/*.log | head -3"
echo "  cat outputs/experiments/multi_sweep_v1_4090/run_status.tsv"
