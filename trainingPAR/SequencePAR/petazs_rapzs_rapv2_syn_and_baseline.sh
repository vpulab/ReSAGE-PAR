


#python train.py PETAzs \
#  --syn_use \
#  --epoch 40 \
#  --synthetic_path /mnt/rhome/paa/pedestrian/dataAugmentationMethods/LORA/diffusers/examples/text_to_image/petazs_lora_32_with_transform_256_192_loss_train_model_21_5_withPETA_reb_forlabeling_por2/ \
#  --syn_pseudolabels_csv generated.csv --seed 42 --syn_use_frac 1.0 --syn_img_subdir generatedImgs/

python train.py RAPzs \
  --syn_use \
  --epoch 40 \
  --synthetic_path /mnt/rhome/paa/pedestrian/dataAugmentationMethods/LORA/diffusers/examples/text_to_image/rapzs_lora_8_with_transform_256_192_loss_train_model_21_4_withRAPv2_reb_forlabeling_por2/ \
  --syn_pseudolabels_csv generated.csv --seed 42 --syn_use_frac 1.0 --syn_img_subdir generatedImgs/

python train.py RAPV2 \
  --syn_use \
  --epoch 40 \
  --synthetic_path /mnt/rhome/paa/pedestrian/dataAugmentationMethods/LORA/diffusers/examples/text_to_image/rapv2_lora_8_with_transform_256_192_loss_train_model_21_reb_forlabeling_por2/ \
  --syn_pseudolabels_csv generated.csv --seed 42 --syn_use_frac 1.0 --syn_img_subdir generatedImgs/


python train.py PETA --epoch 40

python train.py RAPzs --epoch 40

python train.py RAPv2 --epoch 40