export CUDA_VISIBLE_DEVICES=0,1

python -u run.py \
    --task_name pretrain \
    --root_path /media/NAS/1_Datasets/EEG/EEG_benchmark/forecasting/dataset/traffic/ \
    --data_path traffic.csv \
    --model_id Traffic \
    --model SimMTM \
    --data Traffic \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --e_layers 3 \
    --positive_nums 2 \
    --mask_rate 0.5 \
    --enc_in 862 \
    --dec_in 862 \
    --c_out 862 \
    --d_model 128 \
    --d_ff 128 \
    --n_heads 16 \
    --batch_size 2 \
    --dropout 0.2 \
    --train_epochs 50 \
    --temperature 0.02\
    --use_multi_gpu\
    --use_amp
