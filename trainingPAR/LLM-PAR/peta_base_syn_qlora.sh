

export REAL_DATA="/mnt/rhome/paa/pedestrian/Rethinking_of_PAR/data/"
export SYN_DATA="/mnt/rhome/paa/pedestrian/dataAugmentationMethods/LORA/diffusers/examples/text_to_image/"


#python train_qlora.py --dataset PETA --exp peta_baseline --epoch 60 --batchsize 32 --eval_freq 10

SYN_BASE_DIR="$SYN_DATA/peta_lora_32_with_transform_256_192_loss_train_model_21_reb_forlabeling_por2"
python train_qlora.py --dataset PETA \
    --use_synthetic \
    --exp PETA_syn \
    --synthetic_img_dir "$SYN_BASE_DIR/generatedImgs/" \
    --pseudolabels_csv "$SYN_BASE_DIR/generated.csv" \
    --prompts_csv "$SYN_BASE_DIR/prompts.csv" --epoch 60 --eval_freq 30