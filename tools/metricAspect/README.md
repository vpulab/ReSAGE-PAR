# Tools

Analysis and comparison tools for evaluating synthetic vs real pedestrian images.

## Metric Aspect

Compare synthetic and real images using multiple metrics (FID, CFID, CMMD, FD).

### Environment Setup

Create the conda environment:
```bash
conda env create -f ../../environments/tool_metric_aspect.yaml
conda activate tool_metric_aspect
```

### Preparation

Download required third-party helpers:
```bash
./prepareGithubs.sh
```

This script downloads:
- `io_util.py`, `distance.py`, `embedding.py` from `sayakpaul/cmmd-pytorch`
- `fid_score.py` from `mseitzer/pytorch-fid` (modified with additional functions)
- Clones `justin4ai/FD-DINOv2` to `third_party/FD-DINOv2`

### Usage

```bash
python compare_gen_vs_real_metrics.py \
  --CFID --FID --CMMD --FD \
  --dataset RAPzs \
  --nMuestreo 5 \
  --batchSize 96 \
  --pathGen "/path/to/experiment/" \
  --pathReal "/path/to/experiment/condImgs/" \
  --pathSyn "/path/to/experiment/generatedImgs/" \
  --pathGraphic result_rapzs
```

### Parameters

- `--CFID`: Enable CFID calculation (project-specific FID variant)
- `--FID`: Enable Frechet Inception Distance computation
- `--CMMD`: Enable Conditional Maximum Mean Discrepancy computation
- `--FD`: Enable Frechet Distance with DINOv2
- `--dataset <NAME>`: Dataset name (PA100k, PETA, PETAzs, RAPv1, RAPv2, RAPzs)
- `--nMuestreo <N>`: Number of samplings/replicas per attribute or rounds (integer)
- `--batchSize <N>`: Batch size for feature extraction (adjust for GPU/CPU memory)
- `--sampleSizes <N1> <N2> ... <Nn>`: List of sample sizes to evaluate
  - Can include `all` to use full dataset size
  - Example: `--sampleSizes 100 500 1000 all`
  - Default: `[100, 500, 1000, 5000, 10000, 15000, dataset_length]`
- `--pathGen <PATH>`: Base path of experiment (contains `generatedImgs/` and `condImgs/`)
- `--pathReal <PATH>`: Path to real/conditional reference images
- `--pathSyn <PATH>`: Path to synthetic/generated images to evaluate
- `--pathGraphic <NAME_OR_PATH>`: Prefix or path for output graphs/figures

### Outputs

- `<pathGraphic>/FIDScores_output_all.xlsx`: FID scores across sample sizes
- `<pathGraphic>/FIDScores_output_all.csv`: FID scores in CSV format
- `<pathGraphic>/CMMMDScores_output_all.xlsx`: CMMD scores across sample sizes
- `<pathGraphic>/CMMMDScores_output_all.csv`: CMMD scores in CSV format
- `<pathGraphic>/CFIDScores_output_all.xlsx`: CFID scores across sample sizes
- `<pathGraphic>/CFIDScores_output_all.csv`: CFID scores in CSV format
- `<pathGraphic>/FDScores_output_all.xlsx`: FD scores across sample sizes
- `<pathGraphic>/FDScores_output_all.csv`: FD scores in CSV format
- Console output: Computed metric values for each configuration

Each file contains:
- Separate sheets for each sample size or attribute category
- Columns: sample size, metric values, attribute categories (if applicable)

### Supported Metrics

- **FID** (Frechet Inception Distance): Standard measure of distribution similarity
- **CFID** (Conditional FID): Project-specific variant that considers attributes
- **CMMD** (Conditional Maximum Mean Discrepancy): Attribute-conditional distribution comparison
- **FD** (Frechet Distance with DINOv2): Distance metric using DINOv2 features

### Workflow

1. Ensure `prepareGithubs.sh` has been run to fetch dependencies
2. Prepare paths to:
   - Real images (`--pathReal`)
   - Synthetic images (`--pathSyn`)
   - Experiment base directory (`--pathGen`)
3. Run the comparison:
   ```bash
   python compare_gen_vs_real_metrics.py \
     --FID --CMMD \
     --dataset RAPzs \
     --batchSize 96 \
     --pathReal /path/to/real/imgs/ \
     --pathSyn /path/to/synthetic/imgs/ \
     --pathGraphic result_name
   ```
4. Inspect output files for metric values and visualizations

### Notes

- Make sure required dependencies are installed: `torch`, `torchvision`, `pytorch-fid`, `PIL`, `numpy`, etc.
- Batch size should be adjusted based on GPU/CPU memory availability
- For large datasets, consider using smaller `--sampleSizes` to reduce computation time
- The script supports backward compatibility with the original name `comparativeFIDvsCMMD.py`

### Supported Datasets

- `PA100k`
- `PETA`
- `PETAzs`
- `RAPv1`
- `RAPv2`
- `RAPzs`
