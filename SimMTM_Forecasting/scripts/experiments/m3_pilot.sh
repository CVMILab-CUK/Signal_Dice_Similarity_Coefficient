#!/bin/bash
set -e
SEEDS=(42 123 2024 7 1729)
OUTDIR=outputs/m3_pilot
mkdir -p "$OUTDIR"
echo "M3 pilot start $(date -Iseconds)" > "$OUTDIR/pilot.log"
for seed in "${SEEDS[@]}"; do
  echo "[$(date -Iseconds)] seed=$seed pretrain start" >> "$OUTDIR/pilot.log"
  CUDA_VISIBLE_DEVICES=0 TMPDIR=/workspace/tmp /usr/bin/python3 -u run.py \
    --task_name pretrain \
    --root_path /workspace/data/signal/forecasting/ETT-small/ \
    --data_path ETTh1.csv \
    --model_id m3_ETTh1_seed${seed} --model SimMTM --data ETTh1 \
    --features M --seq_len 96 \
    --e_layers 3 --n_heads 16 --d_model 32 --d_ff 64 \
    --positive_nums 3 --mask_rate 0.5 \
    --batch_size 32 --train_epochs 50 \
    --learning_rate 0.001 --dropout 0.1 \
    --loss_mode mse --seed $seed \
    >> "$OUTDIR/seed${seed}.log" 2>&1 || { echo "pretrain failed seed=$seed" >> "$OUTDIR/pilot.log"; continue; }
  echo "[$(date -Iseconds)] seed=$seed finetune+test" >> "$OUTDIR/pilot.log"
  CUDA_VISIBLE_DEVICES=0 TMPDIR=/workspace/tmp /usr/bin/python3 -u run.py \
    --task_name finetune \
    --root_path /workspace/data/signal/forecasting/ETT-small/ \
    --data_path ETTh1.csv \
    --model_id m3_ETTh1_seed${seed} --model SimMTM --data ETTh1 \
    --features M --seq_len 96 --label_len 48 --pred_len 96 \
    --e_layers 3 --n_heads 16 --d_model 32 --d_ff 64 \
    --batch_size 16 --train_epochs 10 \
    --learning_rate 0.0001 --dropout 0.2 \
    --loss_mode mse --seed $seed --is_training 1 \
    >> "$OUTDIR/seed${seed}.log" 2>&1 || { echo "finetune failed seed=$seed" >> "$OUTDIR/pilot.log"; continue; }
  echo "[$(date -Iseconds)] seed=$seed DONE" >> "$OUTDIR/pilot.log"
done
echo "M3 pilot end $(date -Iseconds)" >> "$OUTDIR/pilot.log"
