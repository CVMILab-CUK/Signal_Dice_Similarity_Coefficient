export CUDA_VISIBLE_DEVICES=0,1

python -u run.py \
    --task_name pretrain \
    --root_path /media/NAS/1_Datasets/EEG/EEG_benchmark/forecasting/dataset/weather/ \
    --data_path weather.csv \
    --model_id Weather \
    --model SimMTM \
    --data Weather \
    --features M \
    --seq_len 96 \
    --e_layers 2 \
    --positive_nums 2 \
    --mask_rate 0.5 \
    --enc_in 21 \
    --dec_in 21 \
    --c_out 21 \
    --n_heads 8 \
    --d_model 64 \
    --d_ff 64 \
    --learning_rate 0.001 \
    --batch_size 16 \
    --train_epochs 50\
    --use_multi_gpu\
    --use_amp


