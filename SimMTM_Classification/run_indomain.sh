python ./code/main.py --target_dataset SleepEEG  --fintune_epoch 100
# python ./code/main.py --pretrain_dataset FD-B --target_dataset FD-B --lr 0.0003
# python ./code/main.py --pretrain_dataset Gesture --target_dataset Gesture
python ./code/main.py --pretrain_dataset Epilepsy --target_dataset Epilepsy --finetune_epoch 100
