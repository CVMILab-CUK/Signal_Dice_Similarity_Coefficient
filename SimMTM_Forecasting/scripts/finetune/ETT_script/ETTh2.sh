export CUDA_VISIBLE_DEVICES=0,1

for loss_mode in mse sdsc hybrid; do
    python -u run.py \
        --task_name finetune \
        --is_training 1 \
        --root_path /media/NAS/1_Datasets/EEG/EEG_benchmark/forecasting/dataset/ETT-small/ \
        --data_path ETTh2.csv \
        --model_id ETTh2 \
        --model SimMTM \
        --data ETTh2 \
        --features M \
        --seq_len 96 \
        --label_len 48 \
        --pred_len 96 \
        --e_layers 2 \
        --enc_in 7 \
        --dec_in 7 \
        --c_out 7 \
        --n_heads 8 \
        --d_model 8 \
        --d_ff 32 \
        --dropout 0.4 \
        --head_dropout 0.2 \
        --batch_size 16 \
        --use_multi_gpu \
        --loss_mode $loss_mode
done