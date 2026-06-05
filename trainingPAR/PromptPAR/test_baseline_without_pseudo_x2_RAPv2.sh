

# effect without pseudolabels filtering
#python train.py PETA --use_textprompt --use_div --use_vismask --use_GL --use_mm_former --epoch 40 --syn_use --synthetic_path /mnt/rhome/paa/pedestrian/dataAugmentationMethods/LORA/diffusers/examples/text_to_image/peta_lora_32_with_transform_256_192_loss_train_model_21_forlabeling_4682467166732085907/ --syn_pseudolabels_csv pseudolabeling_notlabeled.csv --seed 42 --syn_use_frac 1.0 --syn_img_subdir generatedImgs/
#python train.py RAPV1 --use_textprompt --use_div --use_vismask --use_GL --use_mm_former --epoch 40 --syn_use --synthetic_path /mnt/rhome/paa/pedestrian/dataAugmentationMethods/LORA/diffusers/examples/text_to_image/rapv1_lora_4_with_transform_256_192_loss_train_model_21_forlabeling_3939659873313549381/ --syn_pseudolabels_csv pseudolabeling_notlabeled.csv --seed 42 --syn_use_frac 1.0 --syn_img_subdir generatedImgs/


python train.py RAPV2 --use_textprompt --use_div \
 --use_vismask --use_GL \
 --use_mm_former --epoch 40 \
 --syn_use \
 --synthetic_path /mnt/rhome/paa/pedestrian/dataAugmentationMethods/LORA/diffusers/examples/text_to_image/rapv2_lora_8_with_transform_256_192_loss_train_model_21_reb_forlabeling_por2/ \
 --syn_pseudolabels_csv generated.csv --seed 42 --syn_use_frac 1.0 --syn_img_subdir generatedImgs/

#python train.py PETAzs --use_textprompt --use_div --use_vismask --use_GL --use_mm_former --epoch 40 --syn_use --synthetic_path /mnt/rhome/paa/pedestrian/dataAugmentationMethods/LORA/diffusers/examples/text_to_image/petazs_lora_32_with_transform_256_192_loss_train_model_21_5_withPETA_forlabeling_2122561973130223505/ --syn_pseudolabels_csv pseudolabeling_notlabeled.csv --seed 42 --syn_use_frac 1.0 --syn_img_subdir generatedImgs/
#python train.py RAPzs --use_textprompt --use_div --use_vismask --use_GL --use_mm_former --epoch 40 --syn_use --synthetic_path /mnt/rhome/paa/pedestrian/dataAugmentationMethods/LORA/diffusers/examples/text_to_image/rapzs_lora_8_with_transform_256_192_loss_train_model_21_4_withRAPv2_forlabeling_8865050951462567614/ --syn_pseudolabels_csv pseudolabeling_notlabeled.csv --seed 42 --syn_use_frac 1.0 --syn_img_subdir generatedImgs/
