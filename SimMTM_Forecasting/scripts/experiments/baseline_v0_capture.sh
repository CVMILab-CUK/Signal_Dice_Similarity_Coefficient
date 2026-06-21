#!/usr/bin/env bash
# Plan v2 Task 0 — Pre-change baseline capture
# Runs mse/sdsc/hybrid on ETTh1 pred_len=96 with seed=2023, saves per-loss JSON.
# Must be run BEFORE any code modification.
set -euo pipefail

ROOT="/root/jeyoung/codes/Signal_Dice_Similarity_Coefficient/SimMTM_Forecasting"
cd "$ROOT"

DATA_ROOT="/workspace/data/signal/forecasting/ETT-small/"
OUT_DIR="${ROOT}/.omc/baseline_v0"
mkdir -p "$OUT_DIR"

# Git hash captured at session start; just verify it's there.
if [[ ! -s "${OUT_DIR}/git_hash.txt" ]]; then
    git rev-parse HEAD > "${OUT_DIR}/git_hash.txt"
fi
echo "Baseline capture at git $(cat ${OUT_DIR}/git_hash.txt)"

for loss_mode in mse sdsc hybrid; do
    echo ""
    echo "============================================================"
    echo "=== BASELINE: loss_mode=${loss_mode} seed=2023 ==="
    echo "============================================================"
    LOG="${OUT_DIR}/${loss_mode}_seed2023.log"

    # Pretrain (50 epochs)
    CUDA_VISIBLE_DEVICES=0 /usr/bin/python3 -u run.py \
        --task_name pretrain \
        --root_path "$DATA_ROOT" \
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
        --loss_mode "$loss_mode" 2>&1 | tee "$LOG"

    # WORKAROUND for path bug (C2): pretrain saves to {data}/ (no loss_mode subdir
    # when alpha=None), but finetune expects {data}/{loss_mode}/.
    # The bug is fixed in US-005. For Task 0 baseline, we relocate manually
    # without modifying source code.
    PRETRAIN_FLAT="${ROOT}/outputs/pretrain_checkpoints/ETTh1"
    PRETRAIN_NESTED="${ROOT}/outputs/pretrain_checkpoints/ETTh1/${loss_mode}"
    if [[ -f "${PRETRAIN_FLAT}/ckpt_best.pth" && ! -f "${PRETRAIN_NESTED}/ckpt_best.pth" ]]; then
        mkdir -p "${PRETRAIN_NESTED}"
        cp "${PRETRAIN_FLAT}/ckpt_best.pth" "${PRETRAIN_NESTED}/ckpt_best.pth"
        # Also copy any periodic checkpoints if present
        for ck in "${PRETRAIN_FLAT}"/ckpt[0-9]*.pth; do
            [[ -f "$ck" ]] && cp "$ck" "${PRETRAIN_NESTED}/"
        done
        echo "Workaround: copied ckpt_best.pth from ${PRETRAIN_FLAT} to ${PRETRAIN_NESTED}"
    fi

    # Finetune + test (using same loss_mode for transfer_checkpoints lookup)
    CUDA_VISIBLE_DEVICES=0 /usr/bin/python3 -u run.py \
        --task_name finetune \
        --is_training 1 \
        --root_path "$DATA_ROOT" \
        --data_path ETTh1.csv \
        --model_id ETTh1 \
        --model SimMTM \
        --data ETTh1 \
        --features M \
        --seq_len 96 --label_len 48 --pred_len 96 \
        --e_layers 3 --enc_in 7 --dec_in 7 --c_out 7 \
        --n_heads 16 --d_model 32 --d_ff 64 \
        --learning_rate 0.0001 \
        --dropout 0.2 \
        --batch_size 16 \
        --loss_mode "$loss_mode" 2>&1 | tee -a "$LOG"

    # Parse the last test result line and save JSON
    # Test output looks like: "96->96, mse:..., mae:..."
    # f.write line: "{seq_len}->{pred_len}, {mse:.3f}, {mae:.3f}"
    # Score file at ./outputs/ETTh1_{loss_mode}_score.txt
    SCORE_FILE="${ROOT}/outputs/ETTh1_${loss_mode}_score.txt"
    if [[ -f "$SCORE_FILE" ]]; then
        LAST=$(tail -n 1 "$SCORE_FILE")
        # Parse: "96->96, 0.385, 0.412"
        MSE=$(echo "$LAST" | awk -F',' '{print $2}' | tr -d ' ')
        MAE=$(echo "$LAST" | awk -F',' '{print $3}' | tr -d ' ')
        /usr/bin/python3 - <<PY
import json
out = {
    "loss_mode": "${loss_mode}",
    "seed": 2023,
    "data": "ETTh1",
    "seq_len": 96,
    "pred_len": 96,
    "mse": float("${MSE}"),
    "mae": float("${MAE}"),
    "git_hash": open("${OUT_DIR}/git_hash.txt").read().strip(),
    "source_file": "${SCORE_FILE}",
}
with open("${OUT_DIR}/${loss_mode}_seed2023.json", "w") as f:
    json.dump(out, f, indent=2)
print(f"Saved baseline JSON for ${loss_mode}: mse={out['mse']:.4f} mae={out['mae']:.4f}")
PY
    else
        echo "WARN: no $SCORE_FILE found after finetune for ${loss_mode}"
    fi
done

echo ""
echo "=== Baseline capture complete ==="
ls -la "${OUT_DIR}/"
