for loss_mode in mse pcc si_snr sdsc; do
    python ./code/main.py  --pretrain_dataset Epilepsy  --target_dataset Epilepsy  --finetune_epoch 100  --training_mode bench_mark --loss_mode $loss_mode
done
