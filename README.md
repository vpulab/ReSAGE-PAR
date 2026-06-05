# 🧪 ReSAGE-PAR: Representational Similarity Assessment for Generative Expansion in Pedestrian Attribute Recognition 🚶‍♂️✨

End-to-end pipeline to **generate dataset-aware synthetic pedestrian images**, **score text–image alignment**, and **produce pseudo-labels** for training Pedestrian Attribute Recognition (PAR) models.

> 🔁 **Stages:** 🖼️ Stage A (Generation) → 📏 Stage B (Scoring) → 🏷️ Stage C (Pseudo-labeling)  
> 📊 Optional: Metric tools to compare **Real vs Synthetic** distributions (FID/CFID/CMMD/FD-DINO)

---

## 📌 Table of Contents
- [✨ Overview](#-overview)
- [⚡ Quick Start](#-quick-start)
- [🧱 Pipeline Stages](#-pipeline-stages)
- [🗂️ Config Files](#️-config-files)
- [🧪 Environments](#-environments)
- [📦 Inputs & Outputs](#-inputs--outputs)
- [🧰 Troubleshooting](#-troubleshooting)

---

## ✨ Overview

This repo supports a **generate–score–autolabel** workflow:
- ✅ **Stage A** generates synthetic images using **Stable Diffusion + LoRA** (optionally with attribute editing).
- ✅ **Stage B** computes **image–text similarity scores** to assess alignment between the synthetic image and its prompt.
- ✅ **Stage C** trains a lightweight **SI-classifier** on real scores and uses it to generate **pseudo-label vectors** for synthetic data.

---

## ⚡ Quick Start

### ✅ Option 1 — Full pipeline (recommended) 🚀

Run everything with YAML configs:

```bash
./wholePipelineWithConfigs.sh
````

This executes:

* 🖼️ Stage A: generation
* 📏 Stage B: scoring (real + synthetic)
* 🏷️ Stage C: pseudo-labeling

---

### 🧩 Option 2 — Run stages individually

#### 🖼️ Stage A — Image generation

```bash
python src/stage_a_generation/run_stage_a.py --config configs/stage_a.yaml
```

#### 📏 Stage B — Scoring (real dataset)

```bash
python -m src.stage_b_scoring.run_stage_b --config configs/stage_b_scores.yaml
```

#### 📏 Stage B — Scoring (synthetic dataset)

```bash
python -m src.stage_b_scoring.run_stage_b --config configs/stage_b_scores_syn.yaml
```

#### 🏷️ Stage C — Train SI-classifier

```bash
python -m src.stage_c_pseudolabeling.run_stage_c --config configs/stage_c_train.yaml
```

#### 🧪 Stage C — Test SI-classifier

```bash
python -m src.stage_c_pseudolabeling.run_stage_c --config configs/stage_c_test.yaml
```

#### 🏷️ Stage C — Label synthetic data

```bash
python -m src.stage_c_pseudolabeling.run_stage_c --config configs/stage_c_labeling_syn.yaml
```

> 💡 **Tip:** Prefer `python -m ...` to avoid relative-import issues.

---

## 🧱 Pipeline Stages

### 🖼️ Stage A — Synthetic Image Generation

📁 `src/stage_a_generation/`
Generate synthetic images using a trained **LoRA** model and prompts derived from attribute vectors.

* Config: `configs/stage_a.yaml`
* Script: `src/stage_a_generation/run_stage_a.py`

---

### 📏 Stage B — Similarity Scoring

📁 `src/stage_b_scoring/`
Compute similarity scores (e.g. BLIP score) for:

* real images + (positive / complemented) prompts (train/test)

* synthetic images + generation prompt

* Configs:

  * `configs/stage_b_scores.yaml` (real)
  * `configs/stage_b_scores_syn.yaml` (synthetic)

---

### 🏷️ Stage C — Pseudo-Labeling

📁 `src/stage_c_pseudolabeling/`
Train/test an SI-classifier on real scores and produce pseudo-labels for synthetic samples.

* Configs:

  * `configs/stage_c_train.yaml`
  * `configs/stage_c_test.yaml`
  * `configs/stage_c_labeling_syn.yaml`

---

### 📊 Tools — Metric Analysis (optional)

📁 `tools/metricAspect/`
Compare distributions between **Real** and **Synthetic** using metrics like:
**FID, CFID, CMMD, FD-DINO**.

---

## 🗂️ Config Files

All configs live in `configs/`:

* `lora.yaml` — LoRA training *(optional / may be commented out in full pipeline)*
* `getMetadata.yaml` — metadata.jsonl generation *(optional)*
* `stage_a.yaml` — generation settings
* `stage_b_scores.yaml` — scoring on real data
* `stage_b_scores_syn.yaml` — scoring on synthetic data
* `stage_c_train.yaml` — classifier training
* `stage_c_test.yaml` — classifier testing
* `stage_c_labeling_syn.yaml` — synthetic pseudo-labeling

### 🧬 Minimal config structure

```yaml
name: "experiment_name"
testing: true

model:
  pretrained_model_name_or_path: "..."

generation:  # or: training, scoring, dataset, etc.
  dataset_name: "RAPzs"

output:
  output_dir: "results/"
```

### 🧰 CLI vs YAML

✅ Run with config:

```bash
python script.py --config configs/stage_a.yaml
```

✅ Override a YAML value:

```bash
python script.py --config configs/stage_a.yaml --height 512
```

> 🔎 Rule: **CLI arguments override YAML values**.

---

## 🧪 Environments

Each stage has its own conda environment for reproducibility:

### 🧬 Create environments

```bash
conda env create -f environments/stage_a.yaml
conda env create -f environments/stage_b.yaml
conda env create -f environments/stage_c.yaml
```

### ✅ Activate

```bash
conda activate stage_a   # LoRA + generation
conda activate stage_b   # scoring
conda activate stage_c   # pseudo-labeling
```

### 📊 Metric tools env (optional)

```bash
conda env create -f environments/tool_metric_aspect.yaml
conda activate tool_metric_aspect
```

---


## 🗃️ Datasets Setup

Before running the pipeline, ensure the real datasets are available on disk. The stages expect three dataset-related paths (which you can override via YAML or CLI in Stages A, B, and C):

- path_dataset: folder with dataset images used by the pipeline
- path_gt: path to the ground-truth pickle (dataset_all.pkl)
- path_gt_img: folder with ground-truth image files (used for GT lookups/metrics)

Defaults vary by dataset. Example defaults for PA100k (as used in configs/getMetadata_custom_paths.yaml):

- path_dataset (images): /mnt/rhome/paa/pedestrian/datasetForFID/PA100k/train/
- path_gt (pickle): /mnt/rhome/paa/pedestrian/Rethinking_of_PAR/data/PA100k/dataset_all.pkl
- path_gt_img (GT images): /mnt/rhome/paa/pedestrian/Rethinking_of_PAR/data/PA100k/data/

Example folder layout (PA100k):

```
/path/
├── datasetdivided/
│   └── PA100k/
│       ├── train/             # path_dataset → images for training split
│       └── test/              
└── data/
    └── PA100k/
      ├── dataset_all.pkl  # path_gt → ground-truth annotations
      └── data/            # path_gt_img → GT image directory
```

You can provide custom paths per dataset in any stage:

- Via YAML: uncomment the `dataset:` block and set `path_dataset`, `path_gt`, `path_gt_img` in the stage config (e.g., configs/stage_a.yaml, configs/stage_b_scores.yaml, configs/stage_c_train.yaml)
- Via CLI: pass `--path_dataset`, `--path_gt`, `--path_gt_img` when running the stage scripts

CLI takes precedence over YAML; if not provided, the code uses dataset-specific defaults.

#### Create train/test splits automatically (PA100k)

If you only have the PA100k ground-truth images and the `dataset_all.pkl`, you can automatically create the split folders using the helper script:

```bash
# Dry-run: preview the actions
python tools/dataset_splitters/create_dataset_split.py \
  --dataset PA100k \
  --pkl /mnt/rhome/paa/pedestrian/Rethinking_of_PAR/data/PA100k/dataset_all.pkl \
  --images_root /mnt/rhome/paa/pedestrian/Rethinking_of_PAR/data/PA100k/data \
  --out_root /mnt/rhome/paa/pedestrian/datasetForFID \
  --mode symlink \
  --dry_run

# Execute: create symlinks into train/ and test/
python tools/dataset_splitters/create_dataset_split.py \
  --dataset PA100k \
  --pkl /mnt/rhome/paa/pedestrian/Rethinking_of_PAR/data/PA100k/dataset_all.pkl \
  --images_root /mnt/rhome/paa/pedestrian/Rethinking_of_PAR/data/PA100k/data \
  --out_root /mnt/rhome/paa/pedestrian/datasetForFID \
  --mode symlink
```

This will create the following structure if it doesn't exist already:

```
/mnt/rhome/paa/pedestrian/datasetForFID/
└── PA100k/
    ├── train/    # or trainval/ for RAPv1
    └── test/
```

You can switch `--mode` to `hardlink` or `copy` if you prefer. Once created, use these paths as `path_dataset` in your stage configs:

- Train: `/mnt/rhome/paa/pedestrian/datasetForFID/PA100k/train/`
- Test: `/mnt/rhome/paa/pedestrian/datasetForFID/PA100k/test/`


### 🗃️ Datasets supported

Use `--dataset` / `dataset_name` with:
`PA100k`, `PETA`, `PETAzs`, `RAPv1`, `RAPv2`, `RAPzs`

---

### 🧾 Metadata for LoRA training (important!)

Before running LoRA training or prompt-based generation, you typically need:

✅ `metadata.jsonl` (captions/prompts for each image)
Optional: include attribute vectors with `--save-vectors`.

Example:

```bash
# Run from repo root (recommended)
PYTHONPATH=. python src/lora_training/getMetadataDataset.py \
  --module customDatasets.RAPzsAll \
  --class RAPzsDatasetAll \
  --pathDataset /path/to/RAPzs/ \
  --num-images 17062 \
  --save-vectors
```

---

## 📦 Inputs & Outputs

### 🖼️ Stage A outputs

Given `--path_syn <OUTPUT_DIR>` (or config equivalent), you should get:

* `<path_syn>/condImgs/` ✅ conditional (real) images
* `<path_syn>/generatedImgs/` ✅ synthetic images
* `<path_syn>/generated.csv` ✅ prompt + filenames (+ optional vectors)

---

### 📏 Stage B outputs

Scores are written under folders like:

* `<dataset>_<prompting>_<score_name>_<strategy>_scores/`

  * `scores_train.xlsx` (+ `.csv`)
  * `scores_test.xlsx`  (+ `.csv`)

Synthetic scoring typically creates:

* `scores_syn.xlsx` / `scores_syn.csv` (depending on config)

---

### 🏷️ Stage C outputs

Artifacts under folders like:

* `<dataset>_<prompting>_<score_name>_<strategy>_<clf_tag>_si/artifacts/`

  * `classifier.pkl`
  * `classifier_tag.txt`
  * `train_predictions.csv`
  * `test_predictions.csv`
  * `pseudolabels_syn.csv`

---

## 🧰 Troubleshooting

### ❌ `attempted relative import with no known parent package`

You’re likely running a file directly like:

```bash
python src/stage_c_pseudolabeling/run_stage_c.py ...
```

✅ Fix: run as a module from repo root:

```bash
PYTHONPATH=. python -m src.stage_c_pseudolabeling.run_stage_c --config configs/stage_c_train.yaml
```

---

### ❌ Pytest import errors (src not found)

Run tests from repo root with:

```bash
PYTHONPATH=. pytest -q
```

---

### ⚠️ “RuntimeWarning: found in sys.modules after import...”

This often happens when mixing `python file.py` and `python -m module`.
✅ Stick to **`python -m ...`** consistently.

---

## 🙌 Notes

* Prefer running from **repo root**.
* Use YAML configs for reproducibility.
* Use `python -m ...` to avoid import issues.

Happy experimenting! 🚀✨

```
::contentReference[oaicite:0]{index=0}
```
