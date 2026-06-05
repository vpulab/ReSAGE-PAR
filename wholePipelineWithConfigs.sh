#!/bin/bash
# Complete pipeline using YAML configuration files instead of CLI arguments
# This script runs all stages: LoRA training, generation, scoring, and pseudo-labeling

eval "$(conda shell.bash hook)"

echo "=========================================="
echo "Starting Synthetic Pseudo-Labeling Pipeline"
echo "=========================================="

## (optional) Stage 0: Generate metadata for dataset
# Uncomment to regenerate metadata with vectors
#echo ""
#echo "Stage 0: Generating metadata..."
python src/lora_training/getMetadataDataset.py --config configs/getMetadata.yaml

## (optional) Stage 0.5: Train LoRA model
# Uncomment to train a new LoRA model
#echo ""
#echo "Stage 0.5: Training LoRA model..."
python src/lora_training/train_lora.py --config configs/lora.yaml

## Stage A: Generation
echo ""
echo "=========================================="
echo "Stage A: Image Generation"
echo "=========================================="
python src/stage_a_generation/run_stage_a.py --config configs/stage_a.yaml

## Stage B: Similarity Indicators (Scoring)
echo ""
echo "=========================================="
echo "Stage B: Similarity Scoring"
echo "=========================================="
conda activate stage_b

# Get scores for real dataset (training the Bayesian classifier)
echo ""
echo "Getting scores for real dataset..."
python -m src.stage_b_scoring.run_stage_b --config configs/stage_b_scores.yaml

# Get scores for synthetic dataset (for pseudo-labeling)
echo ""
echo "Getting scores for synthetic dataset..."
python -m src.stage_b_scoring.run_stage_b --config configs/stage_b_scores_syn.yaml

## Stage C: Pseudo-Labeling
echo ""
echo "=========================================="
echo "Stage C: Pseudo-Labeling"
echo "=========================================="
conda activate stage_c

# Train classifier on real scores
echo ""
echo "Training classifier on real scores..."
python -m src.stage_c_pseudolabeling.run_stage_c --config configs/stage_c_train.yaml

# Test classifier on real scores
echo ""
echo "Testing classifier on real scores..."
python -m src.stage_c_pseudolabeling.run_stage_c --config configs/stage_c_test.yaml

# Label synthetic dataset
echo ""
echo "Labeling synthetic dataset..."
python -m src.stage_c_pseudolabeling.run_stage_c --config configs/stage_c_labeling_syn.yaml

echo ""
echo "=========================================="
echo "Pipeline Complete!"
echo "=========================================="
