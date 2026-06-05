

python train.py RAPV1 --use_textprompt \
 --use_div --use_vismask --use_GL \
  --use_mm_former --epoch 40 \
  --syn_use \
  --synthetic_path /mnt/rhome/paa/pedestrian/dataAugmentationMethods/LORA/diffusers/examples/text_to_image/rapv1_lora_4_with_transform_256_192_loss_train_model_21_reb_forlabeling_por2/ \
  --syn_pseudolabels_csv generated.csv --seed 42 --syn_use_frac 1.0 --syn_img_subdir generatedImgs/


python train.py PETA --use_textprompt --use_div \
  --use_vismask --use_GL --use_mm_former --epoch 40 \
  --syn_use \
  --synthetic_path /mnt/rhome/paa/pedestrian/dataAugmentationMethods/LORA/diffusers/examples/text_to_image/peta_lora_32_with_transform_256_192_loss_train_model_21_reb_forlabeling_por2/ \
  --syn_pseudolabels_csv generated.csv --seed 42 --syn_use_frac 1.0 --syn_img_subdir generatedImgs/


python train.py PETAzs --use_textprompt --use_div --use_vismask \
  --use_GL --use_mm_former --epoch 40 --syn_use \
  --synthetic_path /mnt/rhome/paa/pedestrian/dataAugmentationMethods/LORA/diffusers/examples/text_to_image/petazs_lora_32_with_transform_256_192_loss_train_model_21_5_withPETA_reb_forlabeling_por2/ \
  --syn_pseudolabels_csv generated.csv --seed 42 --syn_use_frac 1.0 --syn_img_subdir generatedImgs/