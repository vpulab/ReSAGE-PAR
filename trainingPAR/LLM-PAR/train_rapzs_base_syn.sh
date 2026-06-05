export REAL_DATA="/mnt/rhome/paa/pedestrian/Rethinking_of_PAR/data/"
export SYN_DATA="/mnt/rhome/paa/pedestrian/dataAugmentationMethods/LORA/diffusers/examples/text_to_image/"


#python train_qlora.py --dataset RAPzs --exp rapzs_baseline --epoch 60 --eval_freq 30

SYN_BASE_DIR="$SYN_DATA/rapzs_lora_8_with_transform_256_192_loss_train_model_21_4_withRAPv2_reb_forlabeling_por2"
python train_qlora_saveCheckpoint.py --dataset RAPzs \
    --use_synthetic \
    --exp RAPzs_syn \
    --synthetic_img_dir "$SYN_BASE_DIR/generatedImgs/" \
    --pseudolabels_csv "$SYN_BASE_DIR/generated.csv" \
    --prompts_csv "$SYN_BASE_DIR/prompts.csv" --epoch 60 --eval_freq 30