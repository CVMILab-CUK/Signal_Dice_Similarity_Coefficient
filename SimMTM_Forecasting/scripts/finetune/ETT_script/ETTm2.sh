export CUDA_VISIBLE_DEVICES=0

for loss_mode in softdtw freeze; do
    python -u run.py \
        --task_name finetune \
        --is_training 1 \
        --root_path /workspace/data/signal/forecasting/ETT-small/ \
        --data_path ETTm2.csv \
        --model_id ETTm2 \
        --model SimMTM \
        --data ETTm2 \
        --features M \
        --seq_len 96 \
        --label_len 48 \
        --pred_len 96 \
        --e_layers 3 \
        --enc_in 7 \
        --dec_in 7 \
        --c_out 7 \
        --n_heads 8 \
        --d_model 8 \
        --d_ff 16 \
        --dropout 0 \
        --batch_size 64\
        --loss_mode $loss_mode
done

