export CUDA_VISIBLE_DEVICES=0

for loss_mode in softdtw freeze; do
    python -u run.py \
        --task_name finetune \
        --is_training 1 \
        --root_path /workspace/data/signal/forecasting/weather/ \
        --data_path weather.csv \
        --model_id Weather \
        --model SimMTM \
        --data Weather \
        --features M \
        --seq_len 96 \
        --label_len 48 \
        --pred_len 96 \
        --e_layers 2 \
        --enc_in 21 \
        --dec_in 21 \
        --c_out 21 \
        --n_heads 8 \
        --d_model 64 \
        --d_ff 64 \
        --batch_size 16\
        --loss_mode $loss_mode\
        --use_amp
done

