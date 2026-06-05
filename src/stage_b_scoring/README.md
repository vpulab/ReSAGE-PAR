# Stage B: Scoring and Sample Selection

Score synthetic and real images using probe prompts to evaluate text-image alignment and attribute consistency.

## Quick Start

**Using configuration files (recommended):**

Get scores for real dataset:
```bash
python -m src.stage_b_scoring.run_stage_b --config configs/stage_b_scores.yaml
```

Get scores for synthetic dataset:
```bash
python -m src.stage_b_scoring.run_stage_b --config configs/stage_b_scores_syn.yaml
```

**Using CLI arguments:**

Real dataset:
```bash
python -m src.stage_b_scoring.run_stage_b \
  --getScores \
  --dataset RAPzs \
  --score_name blip \
  --prompting fixed-rule \
  --strategy identity \
  --n_probes 1 \
  --lora_dir rapzs_lora_4_with_transform_256_192_loss_train_model_21
```

With custom dataset paths:
```bash
python -m src.stage_b_scoring.run_stage_b \
  --config configs/stage_b_scores.yaml \
  --path_dataset "/custom/dataset/" \
  --path_gt "/custom/dataset.pkl" \
  --path_gt_img "/custom/images/"
```

Synthetic dataset:
```bash
python -m src.stage_b_scoring.run_stage_b \
  --getScoresSyn \
  --dataset RAPzs \
  --score_name blip \
  --prompting fixed-rule \
  --strategy identity \
  --n_probes 1 \
  --lora_dir rapzs_lora_4_with_transform_256_192_loss_train_model_21 \
  --syn_csv_path rapzs_lora_4_with_transform_256_192_loss_train_model_21/generated.csv \
  --syn_img_folder rapzs_lora_4_with_transform_256_192_loss_train_model_21/generatedImgs/
```

## Configuration Files

See `configs/stage_b_scores.yaml` and `configs/stage_b_scores_syn.yaml` for examples.

**For real dataset scoring:**
```yaml
getScores: true
getScoresSyn: false

dataset: "RAPzs"
score_name: "blip"
prompting: "fixed-rule"
strategy: "identity"
n_probes: 1

lora_dir: "rapzs_lora_4_with_transform_256_192_loss_train_model_21"

# Optional: Custom dataset paths (uncomment to override defaults)
# path_dataset: "/custom/dataset/"
# path_gt: "/custom/dataset.pkl"
# path_gt_img: "/custom/images/"
```

**For synthetic dataset scoring:**
```yaml
getScores: false
getScoresSyn: true

dataset: "RAPzs"
score_name: "blip"
prompting: "fixed-rule"
strategy: "identity"
n_probes: 1

lora_dir: "rapzs_lora_4_with_transform_256_192_loss_train_model_21"

# Optional: Custom dataset paths (uncomment to override defaults)
# path_dataset: "/custom/dataset/"
# path_gt: "/custom/dataset.pkl"
# path_gt_img: "/custom/images/"

syn_csv_path: "rapzs_lora_4_with_transform_256_192_loss_train_model_21/generated.csv"
syn_img_folder: "rapzs_lora_4_with_transform_256_192_loss_train_model_21/generatedImgs/"
```

## Inputs

- Dataset with images and attribute vectors (from training/test splits)
- Scoring model (BLIP)
- Probe prompt strategy configuration

## Outputs

- `<dataset>_<prompting>_<score_name>_<strategy>_scores/scores_train.xlsx`: Training set scores
  - Sheet `prompting`: Image paths, positive/negative scores, prompts, attribute counts
  - Sheet `sanity_check`: All columns + individual attribute values (interleaved pos/neg)
- `<dataset>_<prompting>_<score_name>_<strategy>_scores/scores_test.xlsx`: Test set scores (same structure)
- `<dataset>_<prompting>_<score_name>_<strategy>_scores/scores_syn.xlsx`: Synthetic dataset scores

## Scoring Models

- `blip`: Uses BLIP Image-Text Matching scores

## Probe Prompt Strategies

- `identity`: Use ground-truth attributes as-is

## Key Parameters

- `--getScores`: Enable score computation for real dataset
- `--getScoresSyn`: Enable score computation for synthetic dataset
- `--dataset`: Dataset name (PA100k, PETA, PETAzs, RAPv1, RAPv2, RAPzs)
- `--score_name`: Scoring model (blip)
- `--prompting`: Prompting type descriptor
- `--strategy`: Probe strategy (identity)
- `--n_probes`: Number of probe prompts per sample
- `--device`: Computing device (cuda/cpu, auto-detected if None)
- `--lora_dir`: Base directory for Stage B/C outputs
- `--path_dataset`: Custom path to dataset images (overrides defaults)
- `--path_gt`: Custom path to ground truth pickle file
- `--path_gt_img`: Custom path to ground truth images folder
- `--syn_csv_path`: Path to synthetic metadata CSV (for `--getScoresSyn`)
- `--syn_img_folder`: Path to synthetic images folder (for `--getScoresSyn`)

## Workflow

1. Load dataset with ground-truth attribute vectors
2. For each sample:
   - Generate positive target (matching GT attributes)
   - Generate negative target (flipped GT attributes)
   - Create probe prompts using specified strategy
   - Compute scores for each probe prompt with the image
3. Save scores, prompts, and metadata to Excel/CSV files
4. Repeat for both train and test splits (real data) or for synthetic data

## Understanding the Output

- **Positive scores**: Measure alignment when prompts match ground-truth attributes (higher is better)
- **Negative scores**: Measure alignment when prompts contradict ground-truth (lower is better)
- **Score difference**: Large positive-negative gap indicates good attribute consistency
- **Sanity check sheet**: Provides per-attribute breakdown for detailed analysis

## Supported Datasets

- `PA100k`
- `PETA`
- `PETAzs`
- `RAPv1`
- `RAPv2`
- `RAPzs`
