export CUDA_VISIBLE_DEVICES=0,1


for points in 10 20 30 40 50; do
    python -u run.py \
        --task_name finetune \
        --root_path /media/NAS/1_Datasets/EEG/EEG_benchmark/forecasting/dataset/traffic/ \
        --data_path traffic.csv \
        --model_id Traffic \
        --model SimMTM \
        --data Traffic \
        --features M \
        --seq_len 96 \
        --label_len 48 \
        --pred_len 96 \
        --e_layers 2 \
        --enc_in 862 \
        --dec_in 862 \
        --c_out 862 \
        --d_model 128 \
        --d_ff 256 \
        --n_heads 16 \
        --batch_size 2 \
        --dropout 0.2\
        --use_multi_gpu \
        --loss_mode mse \
        --transfer_checkpoints ckpt$points.pth\
        --use_amp
done


