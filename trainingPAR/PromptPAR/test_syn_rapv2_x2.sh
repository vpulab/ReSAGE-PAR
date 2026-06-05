

python train.py RAPV2 --use_textprompt --use_div \
 --use_vismask --use_GL \
 --use_mm_former --epoch 40 \
 --syn_use \
 --synthetic_path /mnt/rhome/paa/pedestrian/dataAugmentationMethods/LORA/diffusers/examples/text_to_image/rapv2_lora_8_with_transform_256_192_loss_train_model_21_reb_forlabeling_por2/ \
 --syn_pseudolabels_csv generated.csv --seed 42 --syn_use_frac 1.0 --syn_img_subdir generatedImgs/