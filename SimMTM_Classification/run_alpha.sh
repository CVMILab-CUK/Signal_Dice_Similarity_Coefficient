for alpha in 1 10 100; do
python ./code/main.py --pretrain_dataset Epilepsy --target_dataset Epilepsy --finetune_epoch 100 --alpha $alpha
done
