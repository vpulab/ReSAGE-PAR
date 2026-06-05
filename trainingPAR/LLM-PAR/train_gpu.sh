


export REAL_DATA="/mnt/rhome/paa/pedestrian/Rethinking_of_PAR/data/"
export SYN_DATA="/mnt/rhome/paa/pedestrian/dataAugmentationMethods/LORA/diffusers/examples/text_to_image/"


SYN_BASE_DIR="$SYN_DATA/peta_lora_32_with_transform_256_192_loss_train_model_21_reb_forlabeling_por2"
python train.py --dataset PETA \
    --use_synthetic \
    --exp PETA_syn \
    --synthetic_img_dir "$SYN_BASE_DIR/generatedImgs/" \
    --pseudolabels_csv "$SYN_BASE_DIR/generated.csv" \
    --prompts_csv "$SYN_BASE_DIR/prompts.csv" --epoch 60


SYN_BASE_DIR="$SYN_DATA/pa100k_lora_32_with_transform_256_192_loss_train_model_21_reb_forlabeling_por2"
python train.py --dataset PA100k \
    --use_synthetic \
    --exp PA100k_syn \
    --synthetic_img_dir "$SYN_BASE_DIR/generatedImgs/" \
    --pseudolabels_csv "$SYN_BASE_DIR/generated.csv" \
    --prompts_csv "$SYN_BASE_DIR/prompts.csv" --epoch 60

SYN_BASE_DIR="$SYN_DATA/rapv1_lora_4_with_transform_256_192_loss_train_model_21_reb_forlabeling_por2"
python train.py --dataset RAPv1 \
    --use_synthetic \
    --exp RAPv1_syn \
    --synthetic_img_dir "$SYN_BASE_DIR/generatedImgs/" \
    --pseudolabels_csv "$SYN_BASE_DIR/generated.csv" \
    --prompts_csv "$SYN_BASE_DIR/prompts.csv" --epoch 60

SYN_BASE_DIR="$SYN_DATA/petazs_lora_32_with_transform_256_192_loss_train_model_21_5_withPETA_reb_forlabeling_por2"
python train.py --dataset PETAzs \
    --use_synthetic \
    --exp PETAzs_syn \
    --synthetic_img_dir "$SYN_BASE_DIR/generatedImgs/" \
    --pseudolabels_csv "$SYN_BASE_DIR/generated.csv" \
    --prompts_csv "$SYN_BASE_DIR/prompts.csv" --epoch 60

SYN_BASE_DIR="$SYN_DATA/rapzs_lora_8_with_transform_256_192_loss_train_model_21_4_withRAPv2_reb_forlabeling_por2"
python train.py --dataset RAPzs \
    --use_synthetic \
    --exp RAPzs_syn \
    --synthetic_img_dir "$SYN_BASE_DIR/generatedImgs/" \
    --pseudolabels_csv "$SYN_BASE_DIR/generated.csv" \
    --prompts_csv "$SYN_BASE_DIR/prompts.csv" --epoch 60

SYN_BASE_DIR="$SYN_DATA/rapv2_lora_8_with_transform_256_192_loss_train_model_21_reb_forlabeling_por2"
python train.py --dataset RAPv2 \
    --use_synthetic \
    --exp RAPv2_syn \
    --synthetic_img_dir "$SYN_BASE_DIR/generatedImgs/" \
    --pseudolabels_csv "$SYN_BASE_DIR/generated.csv" \
    --prompts_csv "$SYN_BASE_DIR/prompts.csv" --epoch 60