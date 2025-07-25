export CUDA_VISIBLE_DEVICES=0,1

python -u run.py \
    --task_name pretrain \
    --root_path /media/NAS/1_Datasets/EEG/EEG_benchmark/forecasting/dataset/electricity/ \
    --data_path electricity.csv \
    --model_id ECL \
    --model SimMTM \
    --data ECL \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --e_layers 2 \
    --positive_nums 2 \
    --mask_rate 0.5 \
    --enc_in 321 \
    --dec_in 321 \
    --c_out 321 \
    --d_model 32 \
    --d_ff 64 \
    --n_heads 16 \
    --batch_size 8 \
    --train_epochs 50 \
    --temperature 0.02 \
    --use_multi_gpu
