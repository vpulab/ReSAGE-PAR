export REAL_DATA="/mnt/rhome/paa/pedestrian/Rethinking_of_PAR/data/"
export SYN_DATA="/mnt/rhome/paa/pedestrian/dataAugmentationMethods/LORA/diffusers/examples/text_to_image/"





python train_qlora_saveCheckpoint.py --dataset PA100k --exp pa100k_baseline_test --epoch 60 --eval_freq 30
