#!/usr/bin/env bash
# Plan v2 Task 6 — Full 5-loss x 5-seed sweep on ETTh1.
# Run AFTER US-001..005 code changes and AFTER US-000 baseline capture.
#
# Estimated wall-clock on 1x RTX 6000 Ada: ~50-70 hours total.
#   pretrain ~45 min + finetune ~15 min ~ 1 hour/run x 50 runs = 50 hours,
#   DILATE adds extra (O(T^2) numba).
#
# Resumable: each completed run logs to STATUS_FILE. Re-running this script
# will skip runs whose status is "DONE".
set -uo pipefail

ROOT="/root/jeyoung/codes/Signal_Dice_Similarity_Coefficient/SimMTM_Forecasting"
cd "$ROOT"

DATA_ROOT="/workspace/data/signal/forecasting/ETT-small/"
OUT_DIR="${ROOT}/outputs/experiments/baseline_v1"
STATUS_FILE="${OUT_DIR}/run_status.tsv"
mkdir -p "$OUT_DIR"
[[ -f "$STATUS_FILE" ]] || printf "timestamp\tloss_mode\tseed\tstatus\n" > "$STATUS_FILE"

LOSS_MODES=(mse sdsc hybrid zcr dilate)
SEEDS=(2023 2024 2025 2026 2027)

is_done () {
    grep -qE "^[^\t]+\t$1\t$2\tDONE\$" "$STATUS_FILE" 2>/dev/null
}
mark () {
    printf "%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "$1" "$2" "$3" >> "$STATUS_FILE"
}

total=$(( ${#LOSS_MODES[@]} * ${#SEEDS[@]} ))
done_count=0
for loss_mode in "${LOSS_MODES[@]}"; do
    for seed in "${SEEDS[@]}"; do
        if is_done "$loss_mode" "$seed"; then
            done_count=$(( done_count + 1 ))
            echo "[SKIP] ${loss_mode} seed=${seed} (already DONE)"
            continue
        fi

        echo ""
        echo "==============================================="
        echo "[RUN ${done_count}/$total] loss_mode=${loss_mode} seed=${seed}"
        echo "==============================================="
        RUN_LOG="${OUT_DIR}/${loss_mode}_seed${seed}.log"

        # --- pretrain (50 epochs) ---
        if ! CUDA_VISIBLE_DEVICES=0 /usr/bin/python3 -u run.py \
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
            --loss_mode "$loss_mode" \
            --seed "$seed" 2>&1 | tee "$RUN_LOG"
        then
            mark "$loss_mode" "$seed" "PRETRAIN_FAIL"
            echo "[FAIL] pretrain ${loss_mode} seed=${seed}"
            continue
        fi

        # --- finetune + test ---
        if ! CUDA_VISIBLE_DEVICES=0 /usr/bin/python3 -u run.py \
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
            --loss_mode "$loss_mode" \
            --seed "$seed" 2>&1 | tee -a "$RUN_LOG"
        then
            mark "$loss_mode" "$seed" "FINETUNE_FAIL"
            echo "[FAIL] finetune ${loss_mode} seed=${seed}"
            continue
        fi

        mark "$loss_mode" "$seed" "DONE"
        done_count=$(( done_count + 1 ))
        echo "[DONE] ${loss_mode} seed=${seed} (${done_count}/${total})"
    done
done

echo ""
echo "=== Sweep complete ==="
echo "Total: $total, completed: $done_count"
echo "Status file: $STATUS_FILE"
echo "Run analyze_results.py to produce summary."
