#!/usr/bin/env bash
# Master pipeline: wait for US-000 baseline -> run regression check -> launch US-006 sweep.
# Designed to be nohup'd so it survives user PC handoff / SSH disconnect.
#
# Usage:
#   nohup bash run_full_pipeline.sh > /tmp/full_pipeline.log 2>&1 &
#   echo $! > /tmp/full_pipeline.pid
#
# Resumable: baseline_v1_sweep_ETTh1.sh tracks per-run status via run_status.tsv,
# so re-running this script will skip already-DONE runs.
set -uo pipefail

ROOT="/root/jeyoung/codes/Signal_Dice_Similarity_Coefficient/SimMTM_Forecasting"
cd "$ROOT"

STAGE_LOG="${ROOT}/outputs/experiments/pipeline_stages.tsv"
mkdir -p "${ROOT}/outputs/experiments"
[[ -f "$STAGE_LOG" ]] || printf "timestamp\tstage\tstatus\tnote\n" > "$STAGE_LOG"

mark () {
    printf "%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "$1" "$2" "$3" >> "$STAGE_LOG"
    echo "[$(date -Iseconds)] $1 -> $2 ($3)"
}

# ------------------------------------------------------------------ STAGE 1: wait for baseline
BASELINE_PID_FILE=/tmp/baseline_v0_capture.pid
mark "wait_for_baseline" "START" "PID file=$BASELINE_PID_FILE"

if [[ -f "$BASELINE_PID_FILE" ]]; then
    BASELINE_PID=$(cat "$BASELINE_PID_FILE")
    if kill -0 "$BASELINE_PID" 2>/dev/null; then
        echo "Baseline (PID $BASELINE_PID) is running. Polling until it exits..."
        # Poll every 60s; print a heartbeat
        while kill -0 "$BASELINE_PID" 2>/dev/null; do
            sleep 60
            ELAPSED=$(ps -p "$BASELINE_PID" -o etime= 2>/dev/null | xargs || true)
            EPS_DONE=$(grep -cE "^Epoch:" /tmp/baseline_v0_capture.log 2>/dev/null || true)
            echo "[$(date -Iseconds)] baseline still running, elapsed=${ELAPSED}, epochs_done=${EPS_DONE}/150"
        done
        mark "wait_for_baseline" "DONE" "baseline PID $BASELINE_PID exited"
    else
        mark "wait_for_baseline" "SKIP" "baseline PID $BASELINE_PID not running (already finished)"
    fi
else
    mark "wait_for_baseline" "SKIP" "no baseline PID file; assuming already done"
fi

# ------------------------------------------------------------------ STAGE 2: verify baseline JSONs
mark "verify_baseline_jsons" "START" ""
BASELINE_DIR="${ROOT}/.omc/baseline_v0"
missing=0
for lm in mse sdsc hybrid; do
    if [[ ! -f "${BASELINE_DIR}/${lm}_seed2023.json" ]]; then
        echo "WARN: baseline JSON missing for ${lm}"
        missing=$(( missing + 1 ))
    fi
done
if [[ $missing -gt 0 ]]; then
    mark "verify_baseline_jsons" "WARN" "$missing baselines missing; sweep will run anyway"
else
    mark "verify_baseline_jsons" "OK" "all 3 baselines present"
fi

# ------------------------------------------------------------------ STAGE 3: post-change MSE regression smoke
# Run loss_mode=mse seed=2023 with the NEW code and compare to baseline JSON.
# Skip if a smoke marker already exists (idempotent across restarts).
SMOKE_MARK="${BASELINE_DIR}/post_change_mse_smoke.json"
if [[ ! -f "$SMOKE_MARK" ]]; then
    mark "smoke_post_change_mse" "START" ""
    # Clean prior score file so we can read a fresh test result.
    rm -f "${ROOT}/outputs/ETTh1_mse_score.txt" 2>/dev/null || true
    # Use a separate model_id so it does not collide with the new sweep checkpoints.
    CUDA_VISIBLE_DEVICES=0 /usr/bin/python3 -u run.py \
        --task_name pretrain \
        --root_path /workspace/data/signal/forecasting/ETT-small/ \
        --data_path ETTh1.csv \
        --model_id ETTh1 \
        --model SimMTM \
        --data ETTh1 \
        --features M \
        --seq_len 96 \
        --e_layers 3 --enc_in 7 --dec_in 7 --c_out 7 \
        --n_heads 16 --d_model 32 --d_ff 64 \
        --positive_nums 3 --mask_rate 0.5 \
        --learning_rate 0.001 \
        --batch_size 32 \
        --train_epochs 50 \
        --loss_mode mse --seed 2023 2>&1 | tee "${BASELINE_DIR}/post_change_mse_pretrain.log"

    CUDA_VISIBLE_DEVICES=0 /usr/bin/python3 -u run.py \
        --task_name finetune --is_training 1 \
        --root_path /workspace/data/signal/forecasting/ETT-small/ \
        --data_path ETTh1.csv \
        --model_id ETTh1 \
        --model SimMTM \
        --data ETTh1 \
        --features M \
        --seq_len 96 --label_len 48 --pred_len 96 \
        --e_layers 3 --enc_in 7 --dec_in 7 --c_out 7 \
        --n_heads 16 --d_model 32 --d_ff 64 \
        --learning_rate 0.0001 --dropout 0.2 \
        --batch_size 16 \
        --loss_mode mse --seed 2023 2>&1 | tee -a "${BASELINE_DIR}/post_change_mse_pretrain.log"

    # Compute |Δmse| / mse_baseline and store result
    /usr/bin/python3 - <<'PYEOF' > "$SMOKE_MARK"
import json, os, sys
root = "/root/jeyoung/codes/Signal_Dice_Similarity_Coefficient/SimMTM_Forecasting"
baseline_path = f"{root}/.omc/baseline_v0/mse_seed2023.json"
score_path = f"{root}/outputs/ETTh1_mse_score.txt"
out = {"baseline_path": baseline_path, "score_path": score_path}
try:
    with open(baseline_path) as f:
        base = json.load(f)
    mse_baseline = base["mse"]; mae_baseline = base["mae"]
    with open(score_path) as f:
        last = f.readlines()[-1].strip().split(",")
    mse_new = float(last[1].strip())
    mae_new = float(last[2].strip())
    out["mse_baseline"] = mse_baseline
    out["mse_new"] = mse_new
    out["mae_baseline"] = mae_baseline
    out["mae_new"] = mae_new
    out["rel_mse_diff"] = abs(mse_new - mse_baseline) / max(abs(mse_baseline), 1e-12)
    out["rel_mae_diff"] = abs(mae_new - mae_baseline) / max(abs(mae_baseline), 1e-12)
    out["regression_ok_1e-4"] = (out["rel_mse_diff"] <= 1e-4 and out["rel_mae_diff"] <= 1e-4)
    out["regression_ok_1e-2"] = (out["rel_mse_diff"] <= 1e-2 and out["rel_mae_diff"] <= 1e-2)
except Exception as e:
    out["error"] = str(e)
print(json.dumps(out, indent=2))
PYEOF
    mark "smoke_post_change_mse" "DONE" "see $SMOKE_MARK"
else
    mark "smoke_post_change_mse" "SKIP" "$SMOKE_MARK already exists"
fi

# ------------------------------------------------------------------ STAGE 4: full 5x5 sweep
mark "sweep_v1" "START" ""
bash "${ROOT}/scripts/experiments/baseline_v1_sweep_ETTh1.sh"
mark "sweep_v1" "DONE" ""

# ------------------------------------------------------------------ STAGE 5: analyze
mark "analyze" "START" ""
/usr/bin/python3 "${ROOT}/scripts/experiments/analyze_results.py" \
    --score-dir "${ROOT}/outputs" \
    --out-dir "${ROOT}/outputs/experiments/baseline_v1"
mark "analyze" "DONE" "see outputs/experiments/baseline_v1/results_table.md"

echo ""
echo "================== PIPELINE COMPLETE =================="
echo "Stage log:    $STAGE_LOG"
echo "Sweep status: ${ROOT}/outputs/experiments/baseline_v1/run_status.tsv"
echo "Result table: ${ROOT}/outputs/experiments/baseline_v1/results_table.md"
