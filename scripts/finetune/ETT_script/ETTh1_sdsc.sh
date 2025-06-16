export CUDA_VISIBLE_DEVICES=0,1

for points in 10 20 30 40 50; do
    python -u run.py \
        --task_name finetune \
        --is_training 1 \
        --root_path /media/NAS/1_Datasets/EEG/EEG_benchmark/forecasting/dataset/ETT-small/ \
        --data_path ETTh1.csv \
        --model_id ETTh1 \
        --model SimMTM \
        --data ETTh1 \
        --features M \
        --seq_len 96 \
        --label_len 48 \
        --pred_len 96 \
        --e_layers 3 \
        --enc_in 7 \
        --dec_in 7 \
        --c_out 7 \
        --n_heads 16 \
        --d_model 32 \
        --d_ff 64 \
        --learning_rate 0.0001 \
        --dropout 0.2 \
        --batch_size 16 \
        --use_multi_gpu \
        --loss_mode mse \
        --transfer_checkpoints ckpt$points.pth
done


