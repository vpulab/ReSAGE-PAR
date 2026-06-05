# Stage A: Synthetic Image Generation with LoRA

Generate synthetic pedestrian images using a trained LoRA model, with optional attribute editing and prompt generation from attribute vectors.

## Quick Start

**Using configuration file (recommended):**
```bash
python src/stage_a_generation/run_stage_a.py --config configs/stage_a.yaml
```

**Using CLI arguments:**
```bash
python src/stage_a_generation/run_stage_a.py \
  --path_syn "rapzs_lora_4_with_transform_256_192_loss_train_model_21/" \
  --testing \
  --pretrained_model_name_or_path "../../stablediffusionmodel/..." \
  --output_dir "rapzs_lora_4_with_transform_256_192_loss_train_model_21" \
  --dataset_name RAPzs \
  --prompt_dataset RAPzs \
  --prompt_format_type "fixed-rule" \
  --attribute_policy "gt" \
  --batch_size_memory 5000 \
  --height 256 --width 192
```

**With custom dataset paths:**
```bash
python src/stage_a_generation/run_stage_a.py \
  --config configs/stage_a.yaml \
  --path_dataset "/custom/dataset/images/" \
  --path_gt "/custom/dataset_all.pkl" \
  --path_gt_img "/custom/images/"
```

## Configuration File

See `configs/stage_a.yaml` for a complete example. Key parameters:

```yaml
model:
  pretrained_model_name_or_path: "path/to/model"

dataset:
  # Optional: Custom dataset paths (uncomment to override defaults)
  # path_dataset: "/custom/dataset/images/"
  # path_gt: "/custom/dataset_all.pkl"
  # path_gt_img: "/custom/images/"

generation:
  path_syn: "output/"
  output_dir: "output/"
  dataset_name: "RAPzs"
  height: 512
  width: 512
  attribute_policy: "identity"
  prompt_dataset: "RAPzs"
  prompt_format_type: "fixed-rule"
  batch_size_memory: 5000

testing: true
```

## Inputs

- Trained LoRA model weights (from `train_lora.py`)
- Dataset with `metadata.jsonl` containing image filenames, prompts, and attribute vectors

## Outputs

- `<path_syn>/condImgs/`: Conditional images (`img-{N}.png`)
- `<path_syn>/generatedImgs/`: Generated synthetic images (`img-{N}.png`)
- `<path_syn>/generated.csv`: Metadata CSV with:
  - `file_name`: Image filename
  - `text`: Prompt used
  - `vector_gt_*`: Ground truth attributes
  - `vector_sel_*`: Selected/edited attributes

## Attribute Editing Policies

Control how attribute vectors are transformed before prompt generation:

- `identity`: Use original vector unchanged
- `flip`: Invert all bits (1→0, 0→1)
- `top_k:N`: Keep only top N attribute values
- `threshold:T`: Keep attributes with value ≥ T
- `random_k:K`: Randomly select K attributes to set to 1
- `keep:0,2,5`: Keep original only at specified indices
- `set:1,3,4`: Force specified indices to 1, zero rest

## Supported Datasets

- `PA100k`
- `PETA`
- `PETAzs`
- `RAPv1`
- `RAPv2`
- `RAPzs`

## Key Parameters

- `--pretrained_model_name_or_path`: Base diffusion model
- `--output_dir`: Directory with LoRA weights
- `--dataset_name`: Dataset for loading metadata.jsonl
- `--path_syn`: Output directory for generated images
- `--path_dataset`: Custom path to dataset images (overrides defaults)
- `--path_gt`: Custom path to ground truth pickle file
- `--path_gt_img`: Custom path to ground truth images folder
- `--attribute_policy`: How to transform attributes before generation
- `--prompt_dataset`: Dataset for prompt generation (usually same as `--dataset_name`)
- `--batch_size_memory`: Batch size to manage memory (default: 5000)
- `--height`, `--width`: Output image dimensions
- `--seed`: Random seed for reproducibility
- `--mixed_precision`: Precision for inference (fp16, bf16, no)

## Workflow

1. Load metadata with attribute vectors from dataset's `metadata.jsonl`
2. For each image:
   - Transform attribute vector according to `--attribute_policy`
   - Convert vector to text prompt via dataset's `generatePrompt()`
   - Generate synthetic image using LoRA-enhanced diffusion pipeline
3. Save generated images and metadata to `--path_syn`
4. Processing done in batches to avoid OOM errors

## Notes

- Requires `metadata.jsonl` with attribute vectors (create with `getMetadataDataset.py --save-vectors`)
- `--prompt_dataset` should match the dataset used for metadata
- Attribute editing policies work on loaded vectors
- Policies like `keep` and `set` expect comma-separated index lists without spaces (e.g., `keep:0,2,5`)
