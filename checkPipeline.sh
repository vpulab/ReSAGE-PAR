

eval "$(conda shell.bash hook)"

#####  RAPzs

conda activate stage_b
# get scores real to train the bayesian
python -m src.stage_b_scoring.run_stage_b \
  --getScores \
  --dataset RAPzs \
  --score_name blip \
  --strategy identity \
  --n_probes 1 \
  --prompting fixed-rule \
  --lora_dir /mnt/rhome/paa/pedestrian/dataAugmentationMethods/LORA/diffusers/examples/text_to_image/rapzs_lora_8_with_transform_256_192_loss_train_model_21_4_withRAPv2_forlabeling_8865050951462567614/

# get scores syn to use for psudolabeling
python -m src.stage_b_scoring.run_stage_b \
  --prompting fixed-rule \
  --strategy identity \
  --n_probes 1 \
  --getScoresSyn \
  --dataset RAPzs \
  --score_name blip \
  --lora_dir /mnt/rhome/paa/pedestrian/dataAugmentationMethods/LORA/diffusers/examples/text_to_image/rapzs_lora_8_with_transform_256_192_loss_train_model_21_4_withRAPv2_forlabeling_8865050951462567614/ \
  --syn_csv_path rapzs_lora_4_with_transform_256_192_loss_train_model_21/generated.csv \
  --syn_img_folder /mnt/rhome/paa/pedestrian/dataAugmentationMethods/LORA/diffusers/examples/text_to_image/rapzs_lora_8_with_transform_256_192_loss_train_model_21_4_withRAPv2_forlabeling_8865050951462567614/generatedImgs/ \


conda activate stage_c

# Train on real scores
python -m src.stage_c_pseudolabeling.run_stage_c \
  train \
  --dataset RAPzs \
  --prompting fixed-rule \
  --score_name blip \
  --strategy identity \
  --threshold 0.5 \
  --classifier bayes \
  --bayes_mode gauss \
  --lora_dir /mnt/rhome/paa/pedestrian/dataAugmentationMethods/LORA/diffusers/examples/text_to_image/rapzs_lora_8_with_transform_256_192_loss_train_model_21_4_withRAPv2_forlabeling_8865050951462567614/

# Test on real scores
python -m src.stage_c_pseudolabeling.run_stage_c \
  test \
  --dataset RAPzs \
  --prompting fixed-rule \
  --score_name blip \
  --strategy identity \
  --threshold 0.5 \
  --classifier bayes \
  --bayes_mode gauss \
  --lora_dir /mnt/rhome/paa/pedestrian/dataAugmentationMethods/LORA/diffusers/examples/text_to_image/rapzs_lora_8_with_transform_256_192_loss_train_model_21_4_withRAPv2_forlabeling_8865050951462567614/

# Label synthetic scores
python -m src.stage_c_pseudolabeling.run_stage_c \
  labelingSyn \
  --dataset RAPzs \
  --prompting fixed-rule \
  --score_name blip \
  --strategy identity \
  --threshold 0.5 \
  --classifier bayes \
  --bayes_mode gauss \
  --lora_dir /mnt/rhome/paa/pedestrian/dataAugmentationMethods/LORA/diffusers/examples/text_to_image/rapzs_lora_8_with_transform_256_192_loss_train_model_21_4_withRAPv2_forlabeling_8865050951462567614/




#####  PETAzs

conda activate stage_b
# get scores real to train the bayesian
python -m src.stage_b_scoring.run_stage_b \
  --getScores \
  --dataset PETAzs \
  --score_name blip \
  --strategy identity \
  --n_probes 1 \
  --prompting fixed-rule \
  --lora_dir /mnt/rhome/paa/pedestrian/dataAugmentationMethods/LORA/diffusers/examples/text_to_image/petazs_lora_32_with_transform_256_192_loss_train_model_21_5_withPETA_forlabeling_2122561973130223505/

# get scores syn to use for psudolabeling
python -m src.stage_b_scoring.run_stage_b \
  --prompting fixed-rule \
  --strategy identity \
  --n_probes 1 \
  --getScoresSyn \
  --dataset PETAzs \
  --score_name blip \
  --lora_dir /mnt/rhome/paa/pedestrian/dataAugmentationMethods/LORA/diffusers/examples/text_to_image/petazs_lora_32_with_transform_256_192_loss_train_model_21_5_withPETA_forlabeling_2122561973130223505/ \
  --syn_csv_path /mnt/rhome/paa/pedestrian/dataAugmentationMethods/LORA/diffusers/examples/text_to_image/petazs_lora_32_with_transform_256_192_loss_train_model_21_5_withPETA_forlabeling_2122561973130223505/generated.csv \
  --syn_img_folder /mnt/rhome/paa/pedestrian/dataAugmentationMethods/LORA/diffusers/examples/text_to_image/petazs_lora_32_with_transform_256_192_loss_train_model_21_5_withPETA_forlabeling_2122561973130223505/generatedImgs/ \


conda activate stage_c

# Train on real scores
python -m src.stage_c_pseudolabeling.run_stage_c \
  train \
  --dataset PETAzs \
  --prompting fixed-rule \
  --score_name blip \
  --strategy identity \
  --threshold 0.5 \
  --classifier bayes \
  --bayes_mode gauss \
  --lora_dir /mnt/rhome/paa/pedestrian/dataAugmentationMethods/LORA/diffusers/examples/text_to_image/petazs_lora_32_with_transform_256_192_loss_train_model_21_5_withPETA_forlabeling_2122561973130223505/

# Test on real scores
python -m src.stage_c_pseudolabeling.run_stage_c \
  test \
  --dataset PETAzs \
  --prompting fixed-rule \
  --score_name blip \
  --strategy identity \
  --threshold 0.5 \
  --classifier bayes \
  --bayes_mode gauss \
  --lora_dir /mnt/rhome/paa/pedestrian/dataAugmentationMethods/LORA/diffusers/examples/text_to_image/petazs_lora_32_with_transform_256_192_loss_train_model_21_5_withPETA_forlabeling_2122561973130223505/

# Label synthetic scores
python -m src.stage_c_pseudolabeling.run_stage_c \
  labelingSyn \
  --dataset PETAzs \
  --prompting fixed-rule \
  --score_name blip \
  --strategy identity \
  --threshold 0.5 \
  --classifier bayes \
  --bayes_mode gauss \
  --lora_dir /mnt/rhome/paa/pedestrian/dataAugmentationMethods/LORA/diffusers/examples/text_to_image/petazs_lora_32_with_transform_256_192_loss_train_model_21_5_withPETA_forlabeling_2122561973130223505/



#####  RAPv2

conda activate stage_b
# get scores real to train the bayesian
python -m src.stage_b_scoring.run_stage_b \
  --getScores \
  --dataset RAPv2 \
  --score_name blip \
  --strategy identity \
  --n_probes 1 \
  --prompting fixed-rule \
  --lora_dir /mnt/rhome/paa/pedestrian/dataAugmentationMethods/LORA/diffusers/examples/text_to_image/rapv2_lora_8_with_transform_256_192_loss_train_model_21_forlabeling_2192166418937393404/

# get scores syn to use for psudolabeling
python -m src.stage_b_scoring.run_stage_b \
  --prompting fixed-rule \
  --strategy identity \
  --n_probes 1 \
  --getScoresSyn \
  --dataset RAPv2 \
  --score_name blip \
  --lora_dir /mnt/rhome/paa/pedestrian/dataAugmentationMethods/LORA/diffusers/examples/text_to_image/rapv2_lora_8_with_transform_256_192_loss_train_model_21_forlabeling_2192166418937393404/ \
  --syn_csv_path /mnt/rhome/paa/pedestrian/dataAugmentationMethods/LORA/diffusers/examples/text_to_image/rapv2_lora_8_with_transform_256_192_loss_train_model_21_forlabeling_2192166418937393404/generated.csv \
  --syn_img_folder /mnt/rhome/paa/pedestrian/dataAugmentationMethods/LORA/diffusers/examples/text_to_image/rapv2_lora_8_with_transform_256_192_loss_train_model_21_forlabeling_2192166418937393404/generatedImgs/ \


conda activate stage_c

# Train on real scores
python -m src.stage_c_pseudolabeling.run_stage_c \
  train \
  --dataset RAPv2 \
  --prompting fixed-rule \
  --score_name blip \
  --strategy identity \
  --threshold 0.5 \
  --classifier bayes \
  --bayes_mode gauss \
  --lora_dir /mnt/rhome/paa/pedestrian/dataAugmentationMethods/LORA/diffusers/examples/text_to_image/rapv2_lora_8_with_transform_256_192_loss_train_model_21_forlabeling_2192166418937393404/

# Test on real scores
python -m src.stage_c_pseudolabeling.run_stage_c \
  test \
  --dataset RAPv2 \
  --prompting fixed-rule \
  --score_name blip \
  --strategy identity \
  --threshold 0.5 \
  --classifier bayes \
  --bayes_mode gauss \
  --lora_dir /mnt/rhome/paa/pedestrian/dataAugmentationMethods/LORA/diffusers/examples/text_to_image/rapv2_lora_8_with_transform_256_192_loss_train_model_21_forlabeling_2192166418937393404/

# Label synthetic scores
python -m src.stage_c_pseudolabeling.run_stage_c \
  labelingSyn \
  --dataset RAPv2 \
  --prompting fixed-rule \
  --score_name blip \
  --strategy identity \
  --threshold 0.5 \
  --classifier bayes \
  --bayes_mode gauss \
  --lora_dir /mnt/rhome/paa/pedestrian/dataAugmentationMethods/LORA/diffusers/examples/text_to_image/rapv2_lora_8_with_transform_256_192_loss_train_model_21_forlabeling_2192166418937393404/



#####  RAPv1

conda activate stage_b
# get scores real to train the bayesian
python -m src.stage_b_scoring.run_stage_b \
  --getScores \
  --dataset RAPv1 \
  --score_name blip \
  --strategy identity \
  --n_probes 1 \
  --prompting fixed-rule \
  --lora_dir /mnt/rhome/paa/pedestrian/dataAugmentationMethods/LORA/diffusers/examples/text_to_image/rapv1_lora_4_with_transform_256_192_loss_train_model_21_forlabeling_3939659873313549381/

# get scores syn to use for psudolabeling
python -m src.stage_b_scoring.run_stage_b \
  --prompting fixed-rule \
  --strategy identity \
  --n_probes 1 \
  --getScoresSyn \
  --dataset RAPv1 \
  --score_name blip \
  --lora_dir /mnt/rhome/paa/pedestrian/dataAugmentationMethods/LORA/diffusers/examples/text_to_image/rapv1_lora_4_with_transform_256_192_loss_train_model_21_forlabeling_3939659873313549381/ \
  --syn_csv_path /mnt/rhome/paa/pedestrian/dataAugmentationMethods/LORA/diffusers/examples/text_to_image/rapv1_lora_4_with_transform_256_192_loss_train_model_21_forlabeling_3939659873313549381/generated.csv \
  --syn_img_folder /mnt/rhome/paa/pedestrian/dataAugmentationMethods/LORA/diffusers/examples/text_to_image/rapv1_lora_4_with_transform_256_192_loss_train_model_21_forlabeling_3939659873313549381/generatedImgs/ \


conda activate stage_c

# Train on real scores
python -m src.stage_c_pseudolabeling.run_stage_c \
  train \
  --dataset RAPv1 \
  --prompting fixed-rule \
  --score_name blip \
  --strategy identity \
  --threshold 0.5 \
  --classifier bayes \
  --bayes_mode gauss \
  --lora_dir /mnt/rhome/paa/pedestrian/dataAugmentationMethods/LORA/diffusers/examples/text_to_image/rapv1_lora_4_with_transform_256_192_loss_train_model_21_forlabeling_3939659873313549381/

# Test on real scores
python -m src.stage_c_pseudolabeling.run_stage_c \
  test \
  --dataset RAPv1 \
  --prompting fixed-rule \
  --score_name blip \
  --strategy identity \
  --threshold 0.5 \
  --classifier bayes \
  --bayes_mode gauss \
  --lora_dir /mnt/rhome/paa/pedestrian/dataAugmentationMethods/LORA/diffusers/examples/text_to_image/rapv1_lora_4_with_transform_256_192_loss_train_model_21_forlabeling_3939659873313549381/

# Label synthetic scores
python -m src.stage_c_pseudolabeling.run_stage_c \
  labelingSyn \
  --dataset RAPv1 \
  --prompting fixed-rule \
  --score_name blip \
  --strategy identity \
  --threshold 0.5 \
  --classifier bayes \
  --bayes_mode gauss \
  --lora_dir /mnt/rhome/paa/pedestrian/dataAugmentationMethods/LORA/diffusers/examples/text_to_image/rapv1_lora_4_with_transform_256_192_loss_train_model_21_forlabeling_3939659873313549381/




#####  PETA

conda activate stage_b
# get scores real to train the bayesian
python -m src.stage_b_scoring.run_stage_b \
  --getScores \
  --dataset PETA \
  --score_name blip \
  --strategy identity \
  --n_probes 1 \
  --prompting fixed-rule \
  --lora_dir /mnt/rhome/paa/pedestrian/dataAugmentationMethods/LORA/diffusers/examples/text_to_image/peta_lora_32_with_transform_256_192_loss_train_model_21_forlabeling_4682467166732085907/

# get scores syn to use for psudolabeling
python -m src.stage_b_scoring.run_stage_b \
  --prompting fixed-rule \
  --strategy identity \
  --n_probes 1 \
  --getScoresSyn \
  --dataset PETA \
  --score_name blip \
  --lora_dir /mnt/rhome/paa/pedestrian/dataAugmentationMethods/LORA/diffusers/examples/text_to_image/peta_lora_32_with_transform_256_192_loss_train_model_21_forlabeling_4682467166732085907/ \
  --syn_csv_path /mnt/rhome/paa/pedestrian/dataAugmentationMethods/LORA/diffusers/examples/text_to_image/peta_lora_32_with_transform_256_192_loss_train_model_21_forlabeling_4682467166732085907/generated.csv \
  --syn_img_folder /mnt/rhome/paa/pedestrian/dataAugmentationMethods/LORA/diffusers/examples/text_to_image/peta_lora_32_with_transform_256_192_loss_train_model_21_forlabeling_4682467166732085907/generatedImgs/ \


conda activate stage_c

# Train on real scores
python -m src.stage_c_pseudolabeling.run_stage_c \
  train \
  --dataset PETA \
  --prompting fixed-rule \
  --score_name blip \
  --strategy identity \
  --threshold 0.5 \
  --classifier bayes \
  --bayes_mode gauss \
  --lora_dir /mnt/rhome/paa/pedestrian/dataAugmentationMethods/LORA/diffusers/examples/text_to_image/peta_lora_32_with_transform_256_192_loss_train_model_21_forlabeling_4682467166732085907/

# Test on real scores
python -m src.stage_c_pseudolabeling.run_stage_c \
  test \
  --dataset PETA \
  --prompting fixed-rule \
  --score_name blip \
  --strategy identity \
  --threshold 0.5 \
  --classifier bayes \
  --bayes_mode gauss \
  --lora_dir /mnt/rhome/paa/pedestrian/dataAugmentationMethods/LORA/diffusers/examples/text_to_image/peta_lora_32_with_transform_256_192_loss_train_model_21_forlabeling_4682467166732085907/

# Label synthetic scores
python -m src.stage_c_pseudolabeling.run_stage_c \
  labelingSyn \
  --dataset PETA \
  --prompting fixed-rule \
  --score_name blip \
  --strategy identity \
  --threshold 0.5 \
  --classifier bayes \
  --bayes_mode gauss \
  --lora_dir /mnt/rhome/paa/pedestrian/dataAugmentationMethods/LORA/diffusers/examples/text_to_image/peta_lora_32_with_transform_256_192_loss_train_model_21_forlabeling_4682467166732085907/


#####  PA100k

conda activate stage_b
# get scores real to train the bayesian
python -m src.stage_b_scoring.run_stage_b \
  --getScores \
  --dataset PA100k \
  --score_name blip \
  --strategy identity \
  --n_probes 1 \
  --prompting fixed-rule \
  --lora_dir /mnt/rhome/paa/pedestrian/dataAugmentationMethods/LORA/diffusers/examples/text_to_image/pa100k_lora_32_with_transform_256_192_loss_train_model_21_forlabeling_4791613219522997356/

# get scores syn to use for psudolabeling
python -m src.stage_b_scoring.run_stage_b \
  --prompting fixed-rule \
  --strategy identity \
  --n_probes 1 \
  --getScoresSyn \
  --dataset PA100k \
  --score_name blip \
  --lora_dir /mnt/rhome/paa/pedestrian/dataAugmentationMethods/LORA/diffusers/examples/text_to_image/pa100k_lora_32_with_transform_256_192_loss_train_model_21_forlabeling_4791613219522997356/ \
  --syn_csv_path /mnt/rhome/paa/pedestrian/dataAugmentationMethods/LORA/diffusers/examples/text_to_image/pa100k_lora_32_with_transform_256_192_loss_train_model_21_forlabeling_4791613219522997356/generated.csv \
  --syn_img_folder /mnt/rhome/paa/pedestrian/dataAugmentationMethods/LORA/diffusers/examples/text_to_image/pa100k_lora_32_with_transform_256_192_loss_train_model_21_forlabeling_4791613219522997356/generatedImgs/ \


conda activate stage_c

# Train on real scores
python -m src.stage_c_pseudolabeling.run_stage_c \
  train \
  --dataset PA100k \
  --prompting fixed-rule \
  --score_name blip \
  --strategy identity \
  --threshold 0.5 \
  --classifier bayes \
  --bayes_mode gauss \
  --lora_dir /mnt/rhome/paa/pedestrian/dataAugmentationMethods/LORA/diffusers/examples/text_to_image/pa100k_lora_32_with_transform_256_192_loss_train_model_21_forlabeling_4791613219522997356/

# Test on real scores
python -m src.stage_c_pseudolabeling.run_stage_c \
  test \
  --dataset PA100k \
  --prompting fixed-rule \
  --score_name blip \
  --strategy identity \
  --threshold 0.5 \
  --classifier bayes \
  --bayes_mode gauss \
  --lora_dir /mnt/rhome/paa/pedestrian/dataAugmentationMethods/LORA/diffusers/examples/text_to_image/pa100k_lora_32_with_transform_256_192_loss_train_model_21_forlabeling_4791613219522997356/

# Label synthetic scores
python -m src.stage_c_pseudolabeling.run_stage_c \
  labelingSyn \
  --dataset PA100k \
  --prompting fixed-rule \
  --score_name blip \
  --strategy identity \
  --threshold 0.5 \
  --classifier bayes \
  --bayes_mode gauss \
  --lora_dir /mnt/rhome/paa/pedestrian/dataAugmentationMethods/LORA/diffusers/examples/text_to_image/pa100k_lora_32_with_transform_256_192_loss_train_model_21_forlabeling_4791613219522997356/
