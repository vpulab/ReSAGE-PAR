# LoRA Training for Pedestrian Attribute Recognition

This module contains tools for preparing datasets and training LoRA (Low-Rank Adaptation) models for Stable Diffusion to generate synthetic pedestrian images.

## Overview

The LoRA training pipeline consists of two main steps:

1. **Dataset Preparation** (`getMetadataDataset.py`): Generate metadata file with image prompts and attribute vectors
2. **LoRA Training** (`train_lora.py`): Fine-tune Stable Diffusion model with LoRA adapters

## Quick Start

### Using Configuration Files (Recommended)

```bash
# 1. Generate metadata
python src/lora_training/getMetadataDataset.py --config configs/getMetadata.yaml

# 2. Train LoRA model
python src/lora_training/train_lora.py --config configs/lora.yaml
```

### Using CLI Arguments

```bash
# 1. Generate metadata
python src/lora_training/getMetadataDataset.py \
  --module customDatasets.RAPzsAll \
  --class RAPzsDatasetAll \
  --pathDataset /mnt/rhome/paa/pedestrian/datasetForFID/RAPzs/ \
  --num-images 17062 \
  --save-vectors

# 2. Train LoRA model
python src/lora_training/train_lora.py \
  --pretrained_model_name_or_path "stabilityai/stable-diffusion-2-1" \
  --output_dir "rapzs_lora_4_with_transform_256_192_loss_train_model_21" \
  --dataset_name RAPzs \
  --rank 4 \
  --learning_rate 1e-4 \
  --train_batch_size 92 \
  --num_train_epochs 20 \
  --transform \
  --height 256 \
  --width 192
```

## Step 1: Dataset Preparation (getMetadataDataset.py)

Generate a `metadata.jsonl` file containing image filenames, text prompts, and attribute vectors for training.

### Configuration File Method

**Create/Edit `configs/getMetadata.yaml`:**

```yaml
getMetadata:
  # Required: Dataset configuration
  module: customDatasets.RAPzsAll
  class: RAPzsDatasetAll
  
  # Optional: Number of images to process
  num_images: 17062
  
  # Optional: Output path (defaults to dataset's pathToImages/metadata.jsonl)
  # output: /path/to/metadata.jsonl
  
  # Optional: Custom dataset paths (new, recommended)
  path_dataset: /mnt/rhome/paa/pedestrian/datasetForFID/RAPzs/train/
  path_gt: /mnt/rhome/paa/pedestrian/dataAugmentationMethods/datasets/realOnes/RAPzs_100/dataset_zs_run0.pkl
  path_gt_img: /mnt/rhome/paa/pedestrian/Rethinking_of_PAR/data/RAP2/RAP_dataset/
  
  # Optional: Save attribute vectors for generation
  save_vectors: true
```

**Run:**

```bash
python src/lora_training/getMetadataDataset.py --config configs/getMetadata.yaml
```

### CLI Arguments Method

**Available Arguments:**

- `--module` - Module path for dataset class (e.g., `customDatasets.PA100kAll`)
- `--class` - Dataset class name (e.g., `PA100kDatasetAll`)
- `--num-images` - Number of images to process
- `--output` - Output metadata file path
- `--pathDataset` - Legacy: Override dataset path
- `--path-dataset` - Custom path to dataset images (recommended)
- `--path-gt` - Custom path to ground truth pickle file
- `--path-gt-img` - Custom path to ground truth images folder
- `--seed` - Random seed for reproducibility
- `--save-vectors` - Save attribute vectors in metadata (required for attribute-based generation)

**Example:**

```bash
python src/lora_training/getMetadataDataset.py \
  --module customDatasets.RAPzsAll \
  --class RAPzsDatasetAll \
  --num-images 17062 \
  --save-vectors
```

**With Custom Paths:**

```bash
python src/lora_training/getMetadataDataset.py \
  --module customDatasets.PA100kAll \
  --class PA100kDatasetAll \
  --path-dataset /custom/path/images/ \
  --path-gt /custom/path/dataset.pkl \
  --path-gt-img /custom/path/gt_images/ \
  --save-vectors
```

### Supported Datasets

- **PA100k** - `customDatasets.PA100kAll` / `PA100kDatasetAll`
- **PETA** - `customDatasets.PETAAll` / `PETADatasetAll`
- **PETAzs** - `customDatasets.PETAzsAll` / `PETAzsDatasetAll`
- **RAPv1** - `customDatasets.RAPv1All` / `RAPv1DatasetAll`
- **RAPv2** - `customDatasets.RAPv2All` / `RAPv2DatasetAll`
- **RAPzs** - `customDatasets.RAPzsAll` / `RAPzsDatasetAll`

### Output Format

The generated `metadata.jsonl` contains one JSON object per line:

```json
{"file_name": "image001.png", "text": "a woman from front wearing long sleeve", "vector": [1, 0, 1, ...]}
{"file_name": "image002.png", "text": "there is a man from side with backpack", "vector": [0, 1, 0, ...]}
```

- `file_name` - Image filename
- `text` - Generated text prompt from attributes
- `vector` - Attribute vector (only if `--save-vectors` is used)

## Step 2: LoRA Training (train_lora.py)

Fine-tune a Stable Diffusion model using LoRA adapters on the prepared dataset.

### Configuration File Method

**Create/Edit `configs/lora.yaml`:**

```yaml
# Model configuration
model:
  pretrained_model_name_or_path: "stabilityai/stable-diffusion-2-1"
  revision: null
  variant: null

# Training parameters
training:
  output_dir: "rapzs_lora_4_with_transform_256_192_loss_train_model_21"
  seed: null
  resolution: 512
  height: 256
  width: 192
  train_batch_size: 92
  num_train_epochs: 20
  max_train_steps: null
  gradient_accumulation_steps: 1
  learning_rate: 1.0e-4
  lr_scheduler: "constant"
  lr_warmup_steps: 0
  rank: 4
  transform: true
  validation_epochs: 1
  checkpointing_steps: 500

# Dataset configuration
dataset:
  dataset_name: "RAPzs"
  train_data_dir: null
  caption_column: "text"
  max_train_samples: null
  dataloader_num_workers: 0

# Output and logging
output:
  output_dir: "rapzs_lora_4_with_transform_256_192_loss_train_model_21"
  logging_dir: "logs"
  report_to: "tensorboard"
  use_mlflow: false

# Mixed precision and optimization
mixed_precision: "fp16"
allow_tf32: true
```

**Run:**

```bash
python src/lora_training/train_lora.py --config configs/lora.yaml
```

### CLI Arguments Method

**Key Parameters:**

**Model:**
- `--pretrained_model_name_or_path` - Base Stable Diffusion model (HuggingFace ID or local path)
- `--revision` - Model revision/version
- `--variant` - Model variant (e.g., `fp16`)

**Training:**
- `--output_dir` - Where to save LoRA weights and checkpoints
- `--dataset_name` - Dataset name (PA100k, PETA, PETAzs, RAPv1, RAPv2, RAPzs)
- `--train_batch_size` - Training batch size
- `--num_train_epochs` - Number of training epochs
- `--learning_rate` - Learning rate (default: 1e-4)
- `--rank` - LoRA rank (lower = fewer parameters, default: 4)
- `--transform` - Apply image transforms (resize/crop)
- `--height` - Target image height
- `--width` - Target image width
- `--seed` - Random seed for reproducibility

**Checkpointing:**
- `--checkpointing_steps` - Save checkpoint every N steps
- `--validation_epochs` - Validate every N epochs

**Logging:**
- `--report_to` - Logging destination (`tensorboard`, `wandb`, or `all`)
- `--use_mlflow` - Enable MLflow experiment tracking

**Optimization:**
- `--mixed_precision` - Use mixed precision (`no`, `fp16`, `bf16`)
- `--gradient_accumulation_steps` - Accumulate gradients over N steps
- `--lr_scheduler` - Learning rate scheduler type

**Example (Minimal):**

```bash
python src/lora_training/train_lora.py \
  --pretrained_model_name_or_path "stabilityai/stable-diffusion-2-1" \
  --output_dir "rapzs_lora_4_with_transform_256_192_loss_train_model_21" \
  --dataset_name RAPzs \
  --rank 4 \
  --learning_rate 1e-4 \
  --train_batch_size 92 \
  --num_train_epochs 20 \
  --transform \
  --height 256 \
  --width 192
```

**Example (Full):**

```bash
python src/lora_training/train_lora.py \
  --pretrained_model_name_or_path "stabilityai/stable-diffusion-2-1" \
  --output_dir "rapzs_lora_experiment" \
  --dataset_name RAPzs \
  --train_batch_size 32 \
  --num_train_epochs 10 \
  --learning_rate 1e-4 \
  --rank 4 \
  --transform \
  --height 256 \
  --width 192 \
  --mixed_precision fp16 \
  --gradient_accumulation_steps 4 \
  --checkpointing_steps 500 \
  --validation_epochs 1 \
  --seed 42 \
  --use_mlflow \
  --report_to tensorboard
```

### Output Files

After training, the output directory contains:

- `pytorch_lora_weights.safetensors` - Final trained LoRA weights
- `checkpoint-{N}/` - Intermediate checkpoints (if enabled)
  - `pytorch_lora_weights.safetensors` - LoRA weights at step N
  - `optimizer.bin` - Optimizer state
- `condImgs/` - Validation conditional images
- `generatedImgs/` - Validation generated images
- `generated.csv` - Validation metadata
- `logs/` - TensorBoard logs

## Environment Setup

**Create conda environment:**

```bash
conda env create -f environments/stage_a.yaml
conda activate stage_a
```

The `stage_a` environment includes all dependencies for LoRA training:
- `torch`, `torchvision`
- `transformers`, `diffusers`, `peft`, `accelerate`
- `PIL`, `numpy`, `pandas`

## Custom Dataset Paths

All dataset classes now support custom paths for flexibility:

```bash
python src/lora_training/getMetadataDataset.py \
  --module customDatasets.PA100kAll \
  --class PA100kDatasetAll \
  --path-dataset /custom/dataset/train/ \
  --path-gt /custom/dataset_all.pkl \
  --path-gt-img /custom/images/ \
  --save-vectors
```

Or via config:

```yaml
dataset:
  module: customDatasets.PA100kAll
  class: PA100kDatasetAll
  path_dataset: /custom/dataset/train/
  path_gt: /custom/dataset_all.pkl
  path_gt_img: /custom/images/
```

**Default paths per dataset:**

- **PA100k:** `/mnt/rhome/paa/pedestrian/datasetForFID/PA100k/` + `/mnt/rhome/paa/pedestrian/Rethinking_of_PAR/data/PA100k/`
- **PETA:** `/mnt/rhome/paa/pedestrian/datasetForFID/PETA/` + `/mnt/rhome/paa/pedestrian/Rethinking_of_PAR/data/PETA/`
- **PETAzs:** `/mnt/rhome/paa/pedestrian/datasetForFID/PETAzs/` + `/mnt/rhome/paa/pedestrian/Rethinking_of_PAR/data/PETA/`
- **RAPv1:** `/mnt/rhome/paa/pedestrian/datasetForFID/RAPv1/` + `/mnt/rhome/paa/pedestrian/Rethinking_of_PAR/data/RAP/`
- **RAPv2:** `/mnt/rhome/paa/pedestrian/datasetForFID/RAPv2/` + `/mnt/rhome/paa/pedestrian/Rethinking_of_PAR/data/RAP2/`
- **RAPzs:** `/mnt/rhome/paa/pedestrian/datasetForFID/RAPzs/` + `/mnt/rhome/paa/pedestrian/dataAugmentationMethods/datasets/realOnes/RAPzs_100/`

## Tips and Best Practices

### Dataset Preparation

1. **Always use `--save-vectors`** if you plan to do attribute-based generation in Stage A
2. Check the generated `metadata.jsonl` to verify prompts are correct
3. Use `--num-images` to test with a smaller subset first

### LoRA Training

1. **Start with rank=4** for most cases (good balance between quality and speed)
2. **Use `--transform`** if your dataset images need resizing to match model resolution
3. **Enable checkpointing** (`--checkpointing_steps 500`) to save intermediate models
4. **Monitor training:**
   - Use TensorBoard: `tensorboard --logdir logs/`
   - Or MLflow: `mlflow ui` (if `--use_mlflow` enabled)
5. **Batch size tuning:**
   - Reduce if you get OOM errors
   - Use `--gradient_accumulation_steps` to simulate larger batches
6. **Mixed precision (`fp16`)** speeds up training significantly on modern GPUs

### Troubleshooting

**Out of Memory:**
- Reduce `--train_batch_size`
- Lower `--resolution`, `--height`, or `--width`
- Increase `--gradient_accumulation_steps`

**Poor quality generations:**
- Increase `--rank` (try 8 or 16)
- Train for more epochs
- Check that `--height` and `--width` match your target generation size
- Verify `metadata.jsonl` prompts are descriptive

**Slow training:**
- Use `--mixed_precision fp16`
- Reduce `--validation_epochs` frequency
- Disable MLflow if not needed

## Next Steps

After training, use the LoRA weights in Stage A for synthetic image generation:

```bash
python src/stage_a_generation/run_stage_a.py \
  --output_dir "rapzs_lora_4_with_transform_256_192_loss_train_model_21" \
  --dataset_name RAPzs \
  --height 256 --width 192
```

See [src/stage_a_generation/README.md](../stage_a_generation/README.md) for details.
