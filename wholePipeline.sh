
eval "$(conda shell.bash hook)"

## (required for lora training) Previous to stage a
#python src/lora_training/getMetadataDataset.py --module customDatasets.RAPzsAll --class RAPzsDatasetAll --pathDataset /mnt/rhome/paa/pedestrian/datasetForFID/RAPzs/ --num-images 17062 --save-vectors

#python src/lora_training/train_lora.py \
#  --pretrained_model_name_or_path "../../stablediffusionmodel/models--stabilityai--stable-diffusion-2-1/snapshots/5cae40e6a2745ae2b01ad92ae5043f95f23644d6/" \
#  --output_dir "rapzs_lora_4_with_transform_256_192_loss_train_model_21" \
#  --dataset_name RAPzs \
#  --rank 4 \
#  --learning_rate 1e-4 \
#  --train_batch_size 92 \
#  --num_train_epochs 20 \
#  --transform \
#  --height 256 \
#  --width 192

## stage a: generation
#conda activate stage_a

python src/stage_a_generation/run_stage_a.py \
  --path_syn "rapzs_lora_4_with_transform_256_192_loss_train_model_21/" \
  --testing \
  --pretrained_model_name_or_path "../../stablediffusionmodel/models--stabilityai--stable-diffusion-2-1/snapshots/5cae40e6a2745ae2b01ad92ae5043f95f23644d6/" \
  --output_dir "rapzs_lora_4_with_transform_256_192_loss_train_model_21" \
  --dataset_name RAPzs \
  --prompt_dataset RAPzs \
  --prompt_format_type "fixed-rule" \
  --attribute_policy "gt" \
  --batch_size_memory 5000 \
  --height 256 --width 192


## stage b: similarity indicators

conda activate stage_b

# get scores real to train the bayesian
python -m src.stage_b_scoring.run_stage_b \
  --getScores \
  --dataset RAPzs \
  --score_name blip \
  --strategy identity \
  --n_probes 1 \
  --prompting fixed-rule \
  --lora_dir rapzs_lora_4_with_transform_256_192_loss_train_model_21

# get scores syn to use for psudolabeling
python -m src.stage_b_scoring.run_stage_b \
  --prompting fixed-rule \
  --strategy identity \
  --n_probes 1 \
  --getScoresSyn \
  --dataset RAPzs \
  --score_name blip \
  --lora_dir rapzs_lora_4_with_transform_256_192_loss_train_model_21 \
  --syn_csv_path rapzs_lora_4_with_transform_256_192_loss_train_model_21/generated.csv \
  --syn_img_folder rapzs_lora_4_with_transform_256_192_loss_train_model_21/generatedImgs/ \

## stage c: psudolabeling

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
  --lora_dir rapzs_lora_4_with_transform_256_192_loss_train_model_21

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
  --lora_dir rapzs_lora_4_with_transform_256_192_loss_train_model_21

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
  --lora_dir rapzs_lora_4_with_transform_256_192_loss_train_model_21