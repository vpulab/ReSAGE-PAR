"""Runner for Stage B: scoring and selecting samples using probe prompts."""

import argparse
import os
import sys
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Sequence
from PIL import Image
from tqdm import tqdm

try:
    import yaml
except ImportError:
    yaml = None

# Import our modules
from .text_image_representational_similarity import TextImageRepresentationalSimilarity
from .probe_prompt_strategy import ProbePromptStrategy

# Import PromptGenerator
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stage_a_generation"))
from generation_prompt_formatting import PromptGenerator


def get_dataset(dataset_name: str, split: str = "train", path_dataset: str | None = None, path_gt: str | None = None, path_gt_img: str | None = None):
    """Load dataset by name and split.
    
    Args:
        dataset_name: Name of dataset (PA100k, PETA, PETAzs, RAPv1, RAPv2, RAPzs)
        split: 'train' or 'test'
        path_dataset: Optional custom path to images
        path_gt: Optional custom path to ground truth pickle
        path_gt_img: Optional custom path to gt images
    
    Returns:
        Dataset instance
    """
    # Add src directory to path to enable imports
    src_path = os.path.join(os.path.dirname(__file__), "..", "..")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    
    if dataset_name == "PA100k":
        from src.lora_training.customDatasets.datasetPA100kAll import PA100kDatasetAll
        return PA100kDatasetAll(split=split, path_dataset=path_dataset, path_gt=path_gt, path_gt_img=path_gt_img)
    elif dataset_name == "PETA":
        from src.lora_training.customDatasets.datasetPETAAll import PETADatasetAll
        return PETADatasetAll(split=split, path_dataset=path_dataset, path_gt=path_gt, path_gt_img=path_gt_img)
    elif dataset_name == "PETAzs":
        from src.lora_training.customDatasets.datasetPETAzsAll import PETAzsDatasetAll
        return PETAzsDatasetAll(split=split, path_dataset=path_dataset, path_gt=path_gt, path_gt_img=path_gt_img)
    elif dataset_name == "RAPv1":
        from src.lora_training.customDatasets.datasetRAPv1All import RAPv1DatasetAll
        return RAPv1DatasetAll(split=split, path_dataset=path_dataset, path_gt=path_gt, path_gt_img=path_gt_img)
    elif dataset_name == "RAPv2":
        from src.lora_training.customDatasets.datasetRAPv2All import RAPv2DatasetAll
        return RAPv2DatasetAll(split=split, path_dataset=path_dataset, path_gt=path_gt, path_gt_img=path_gt_img)
    elif dataset_name == "RAPzs":
        from src.lora_training.customDatasets.datasetRAPzsAll import RAPzsDatasetAll
        return RAPzsDatasetAll(split=split, path_dataset=path_dataset, path_gt=path_gt, path_gt_img=path_gt_img)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def create_prompting(dataset, prompting_type: str, dataset_name: str, strategy: str = None) -> ProbePromptStrategy:
    """Create prompting strategy instance.
    
    Args:
        dataset: Dataset instance with generatePrompt method
        prompting_type: Type of prompting strategy
        dataset_name: Name of the dataset for PromptGenerator
    
    Returns:
        ProbePromptStrategy instance
    """
    # Create PromptGenerator for the dataset
    prompt_generator = None
    try:
        prompt_generator = PromptGenerator(type=prompting_type, dataset=dataset_name)
    except Exception as e:
        print(f"Warning: Could not create PromptGenerator: {e}")
        # Fallback to dataset's generatePrompt method if available
        prompt_generator = dataset if hasattr(dataset, 'generatePrompt') else None
    
    return ProbePromptStrategy(prompt_generator=prompt_generator, strategy=strategy, dataset_name=dataset_name)


def _build_experiment_path(base_dir: str | None, dataset: str, prompting: str, score_name: str, strategy: str) -> str:
    """Return absolute path for experiment outputs under the given base directory."""
    root = os.path.abspath(base_dir) if base_dir else os.path.abspath(".")
    return os.path.join(root, f"{dataset}_{prompting}_{score_name}_{strategy}_scores")


def getScoresTraining(dataset, score_model: TextImageRepresentationalSimilarity, 
                      prompting: ProbePromptStrategy, n_probes: int = 5,
                      strategy: str = "interpolate:3") -> Tuple:
    """Compute scores for training set.
    
    Args:
        dataset: Dataset instance
        score_model: Scoring model instance
        prompting: Prompting strategy instance
        n_probes: Number of probes to generate
        strategy: Prompting strategy to use
    
    Returns:
        Tuple of (imgpaths, pos_scores, neg_scores, prompt_pos, prompt_neg,
                 num_att_pos, num_att_neg, vector_pos, vector_neg)
    """
    imgpaths = []
    pos_scores = []
    neg_scores = []
    prompt_pos = []
    prompt_neg = []
    num_att_pos = []
    num_att_neg = []
    vector_pos = []
    vector_neg = []
    
    print(f"Computing scores for {len(dataset)} training samples...")
    
    for idx in tqdm(range(len(dataset))):

        sample = dataset[idx]

        # Extract image and attributes
        img = sample[dataset.idxImgRGB]
        gt_vector = sample[dataset.idxVector]
        img_path = sample[dataset.idxImgPath]
        
        gt_vector = np.array(gt_vector) 
        selected_vector = gt_vector

        if img is None or len(gt_vector) == 0:
            continue
        
        
        
        # Generate positive (same as GT) and negative (flipped) target vectors
        target_pos = selected_vector.copy()
        
        # Generate probe prompts for positive case
        probes_pos, probes_neg = prompting.generate_probes(
            gt_vector=gt_vector,
            target_vector=target_pos,
            n_probes=n_probes,
            strategy=strategy
        )
        
        if probes_pos is None or len(probes_pos) == 0:
            print(f"Warning: No positive probes generated for sample {idx}, skipping.")
            continue
        
        imgpaths.append(img_path)

        # Extract target_neg vector from first negative probe
        target_neg = probes_neg[0]['vector']
        
        # Compute scores for positive probes
        prompts_pos_list = [p['prompt'] for p in probes_pos]
        result_pos = score_model.score_prompts_images(prompts_pos_list, img)
        scores_pos = result_pos['scores']
        
        # Compute scores for negative probes 
        prompts_neg_list = [p['prompt'] for p in probes_neg]
        result_neg = score_model.score_prompts_images(prompts_neg_list, img)
        scores_neg = result_neg['scores']
        
        # Store results
        pos_scores.append(scores_pos.tolist())
        neg_scores.append(scores_neg.tolist())
        prompt_pos.append(prompts_pos_list)
        prompt_neg.append(prompts_neg_list)
        num_att_pos.append(dataset.getNumberOfAttributesFromVector(target_pos))
        num_att_neg.append(dataset.getNumberOfAttributesFromVector(target_neg))
        vector_pos.append(target_pos.tolist())
        vector_neg.append(target_neg.tolist())
    
    return (imgpaths, pos_scores, neg_scores, prompt_pos, prompt_neg, 
            num_att_pos, num_att_neg, vector_pos, vector_neg)


def getScoresTesting(dataset, score_model: TextImageRepresentationalSimilarity,
                     prompting: ProbePromptStrategy, n_probes: int = 5,
                     strategy: str = "identity") -> Tuple:
    """Compute scores for test set (same as training)."""
    return getScoresTraining(dataset, score_model, prompting, n_probes, strategy)


def save_scores_xlsx(
    dataset,
    pos: Sequence[float],
    neg: Sequence[float],
    img_paths: Sequence[str] = None,
    out_dir: str = ".",
    train: bool = True,
    is_syn: bool = False,
    prompt_pos: Sequence[str] = None,
    prompt_neg: Sequence[str] = None,
    num_attr_pos: Sequence[int] = None,
    num_attr_neg: Sequence[int] = None,
    vector_pos=None,   # shape: (len(pos),  num_attrs) aligned with dataset.listAllAttrib()
    vector_neg=None,   # shape: (len(neg),  num_attrs) aligned with dataset.listAllAttrib()
) -> str:
    """
    Excel with two sheets:
      - 'prompting': imgpath, pos, neg, prompt_pos, prompt_neg, num_attr_pos, num_attr_neg
      - 'sanity_check': same base columns + for EVERY dataset attribute (dataset.listAllAttrib),
                        two FLAT columns '<attr>_pos' and '<attr>_neg' filled from vector_pos / vector_neg.
    """
    os.makedirs(out_dir, exist_ok=True)
    if is_syn:
        fname = "scores_syn.xlsx"
    else:
        fname = "scores_train.xlsx" if train else "scores_test.xlsx"
    fpath = os.path.join(out_dir, fname)

    # --- base alignment ---
    pos_len = len(pos) if pos is not None else 0
    neg_len = len(neg) if neg is not None else 0
    img_len = len(img_paths) if img_paths is not None else 0
    n = max(pos_len, neg_len, img_len)

    imgspaths = pd.Series(img_paths, dtype="string", name="imgpath").reindex(range(n)) if img_paths is not None \
                else pd.Series([pd.NA]*n, dtype="string", name="imgpath")
    
    # Keep scores as vectors (lists) to allow multiple indicators; wrap scalars into 1-length lists
    pos_values = [[p] if not isinstance(p, list) else p for p in pos] if pos is not None else None
    neg_values = [[n] if not isinstance(n, list) else n for n in neg] if neg is not None else None
    
    pos_s = pd.Series(pos_values, dtype=object, name="pos").reindex(range(n))
    neg_s = pd.Series(neg_values, dtype=object, name="neg").reindex(range(n))

    # Extract only the prompt text (first prompt from each list, removing strategy info)
    def extract_prompt(prompt_list):
        if isinstance(prompt_list, list) and len(prompt_list) > 0:
            # Get first prompt and remove strategy suffix if present
            first_prompt = prompt_list[0]
            # Remove strategy info like "[strategy=identity 1/1]"
            if '[strategy=' in first_prompt:
                first_prompt = first_prompt[:first_prompt.rfind('[')].strip()
            return first_prompt
        return str(prompt_list) if prompt_list else None
    
    prompt_pos_str = [extract_prompt(p) for p in prompt_pos] if prompt_pos is not None else None
    prompt_neg_str = [extract_prompt(n) for n in prompt_neg] if prompt_neg is not None else None
    
    prompt_pos_s = pd.Series(prompt_pos_str, dtype="string", name="prompt_pos").reindex(range(n)) if prompt_pos_str is not None \
                   else pd.Series([pd.NA]*n, dtype="string", name="prompt_pos")
    prompt_neg_s = pd.Series(prompt_neg_str, dtype="string", name="prompt_neg").reindex(range(n)) if prompt_neg_str is not None \
                   else pd.Series([pd.NA]*n, dtype="string", name="prompt_neg")

    num_attr_pos_s = pd.Series(num_attr_pos, dtype=pd.Int16Dtype(), name="num_attr_pos").reindex(range(n)) if num_attr_pos is not None \
                     else pd.Series([pd.NA]*n, dtype=pd.Int16Dtype(), name="num_attr_pos")
    num_attr_neg_s = pd.Series(num_attr_neg, dtype=pd.Int16Dtype(), name="num_attr_neg").reindex(range(n)) if num_attr_neg is not None \
                     else pd.Series([pd.NA]*n, dtype=pd.Int16Dtype(), name="num_attr_neg")

    prompting_df = pd.concat(
        [imgspaths, pos_s, neg_s, prompt_pos_s, prompt_neg_s, num_attr_pos_s, num_attr_neg_s],
        axis=1
    )

    # --- attribute names (aligned with vectors) ---
    try:
        attr_names = [name for name in dataset.listAllAttrib]
    except Exception:
        attr_names = None

    # Coerce vectors
    vp = None if vector_pos is None else np.asarray(vector_pos)
    vn = None if vector_neg is None else np.asarray(vector_neg)

    # Infer number of attributes if needed
    def _width(a):
        if a is None: return None
        a = np.asarray(a)
        if a.ndim == 2: return a.shape[1]
        if a.ndim == 1: return a.shape[0]
        return None

    if not attr_names:
        k = _width(vp) or _width(vn) or 0
        attr_names = [f"attr_{i}" for i in range(int(k))]

    # ---------- robust vector coercion ----------
    def _to_2d_matrix(vec, rows_expected: int, cols_expected: int | None):
        if vec is None:
            return None
        a = np.asarray(vec, dtype=object)

        # Handle empty array early
        if a.size == 0 or rows_expected == 0:
            if cols_expected is None:
                cols_expected = 0
            return np.empty((rows_expected, cols_expected), dtype=float)

        # List of per-sample vectors -> stack
        if a.ndim == 1 and a.size == rows_expected and a.size > 0 and isinstance(a[0], (list, tuple, np.ndarray)):
            a = np.vstack([np.asarray(v).ravel() for v in a])

        # Single vector -> broadcast
        if a.ndim == 1:
            if cols_expected is None:
                cols_expected = int(a.size)
            a = np.broadcast_to(np.asarray(a, dtype=float), (rows_expected, cols_expected))
        elif a.ndim == 2:
            a = np.asarray(a, dtype=float)
        else:
            a = np.squeeze(a)
            return _to_2d_matrix(a, rows_expected, cols_expected)

        # Fix possible transpose
        if cols_expected is not None and a.shape == (cols_expected, rows_expected):
            a = a.T

        # Trim/pad
        r, c = a.shape[:2]
        if cols_expected is None:
            cols_expected = c
        if r < rows_expected:
            a = np.vstack([a, np.full((rows_expected - r, c), np.nan)])
        if c < cols_expected:
            a = np.hstack([a, np.full((a.shape[0], cols_expected - c), np.nan)])
        return a[:rows_expected, :cols_expected]

    vp = _to_2d_matrix(vector_pos, pos_len, len(attr_names) if attr_names else None) if vector_pos is not None else None
    vn = _to_2d_matrix(vector_neg, neg_len, len(attr_names) if attr_names else None) if vector_neg is not None else None

    if not attr_names:
        k = (vp.shape[1] if vp is not None else None) or (vn.shape[1] if vn is not None else 0)
        attr_names = [f"attr_{i}" for i in range(int(k))]

    num_attrs = len(attr_names)

    # ---------- build attribute columns interleaved: <attr>_pos, <attr>_neg ----------
    # Prepare empty DataFrame with all expected columns
    interleaved_cols = []
    for a in attr_names:
        interleaved_cols += [f"{a}_pos", f"{a}_neg"]

    attr_df = pd.DataFrame(index=range(n), columns=interleaved_cols, dtype="Int64")

    # Fill pos values at the top rows (row-aligned)
    if vp is not None and vp.size:
        tmp_pos = pd.DataFrame(vp[:pos_len, :num_attrs], columns=[f"{a}_pos" for a in attr_names])
        tmp_pos = tmp_pos.round().clip(lower=0).astype("Int64")
        attr_df.loc[:pos_len-1, tmp_pos.columns] = tmp_pos.values

    # Fill neg values at the top rows (row-aligned)
    if vn is not None and vn.size:
        tmp_neg = pd.DataFrame(vn[:neg_len, :num_attrs], columns=[f"{a}_neg" for a in attr_names])
        tmp_neg = tmp_neg.round().clip(lower=0).astype("Int64")
        attr_df.loc[:neg_len-1, tmp_neg.columns] = tmp_neg.values

    # Ensure interleaved order already (attr_df was created with that order)

    # ---------- write Excel ----------
    sanity_df = pd.concat([prompting_df, attr_df], axis=1)

    with pd.ExcelWriter(fpath, engine="openpyxl") as xlw:
        prompting_df.to_excel(xlw, index=False, sheet_name="prompting")
        sanity_df.to_excel(xlw, index=False, sheet_name="sanity_check")

    
    print(f"Saved scores to {fpath}")




def run_stage_b(config_path: str):
    print(f"[stage_b] Running Stage B (scoring) with config: {config_path}")
    # TODO: implement scoring / selection logic


def run_get_scores_real(args):
    """Run score computation for both train and test sets on real datasets."""
    print(f"[stage_b] Computing scores for REAL dataset with:")
    print(f"  Dataset: {args.dataset}")
    print(f"  Score type: {args.score_name}")
    print(f"  Prompting: {args.prompting}")
    print(f"  Strategy: {args.strategy}")
    print(f"  N probes: {args.n_probes}")
    if args.lora_dir:
        print(f"  LoRA run dir: {args.lora_dir}")

    
    # Initialize scoring model
    print(f"\nInitializing {args.score_name} scoring model...")
    score_model = TextImageRepresentationalSimilarity(
        score_name=args.score_name,
        device=args.device
    )
    
    # Process training set
    print("\n=== Processing Training Set ===")
    dataset_train = get_dataset(args.dataset, "train", getattr(args, "path_dataset", None), getattr(args, "path_gt", None), getattr(args, "path_gt_img", None))
    prompting_train = create_prompting(dataset_train, args.prompting, args.dataset, args.strategy)
    experiment_path = _build_experiment_path(args.lora_dir, args.dataset, args.prompting, args.score_name, args.strategy)
    
    imgpaths, pos_scores, neg_scores, prompt_pos, prompt_neg, num_att_pos, num_att_neg, vector_pos, vector_neg = \
        getScoresTraining(
            dataset=dataset_train,
            score_model=score_model,
            prompting=prompting_train,
            n_probes=args.n_probes,
            strategy=args.strategy
        )
    
    save_scores_xlsx(
        dataset_train, pos_scores, neg_scores, imgpaths, experiment_path, train=True,
        prompt_pos=prompt_pos, prompt_neg=prompt_neg,
        num_attr_pos=num_att_pos, num_attr_neg=num_att_neg,
        vector_pos=vector_pos, vector_neg=vector_neg
    )
    
    # Process test set
    print("\n=== Processing Test Set ===")
    dataset_test = get_dataset(args.dataset, "test", getattr(args, "path_dataset", None), getattr(args, "path_gt", None), getattr(args, "path_gt_img", None))
    prompting_test = create_prompting(dataset_test, args.prompting, args.dataset, args.strategy)
    
    print(args.strategy)

    imgpaths, pos_scores, neg_scores, prompt_pos, prompt_neg, num_att_pos, num_att_neg, vector_pos, vector_neg = \
        getScoresTesting(
            dataset=dataset_test,
            score_model=score_model,
            prompting=prompting_test,
            n_probes=args.n_probes,
            strategy=args.strategy
        )
    
    save_scores_xlsx(
        dataset_test, pos_scores, neg_scores, imgpaths, experiment_path, train=False,
        prompt_pos=prompt_pos, prompt_neg=prompt_neg,
        num_attr_pos=num_att_pos, num_attr_neg=num_att_neg,
        vector_pos=vector_pos, vector_neg=vector_neg
    )
    
    print(f"\n✓ Scoring complete! Results saved to: {experiment_path}")


def run_get_scores_syn(args):
    """Run score computation for synthetic dataset from CSV and genImg folder."""
    print(f"[stage_b] Computing scores for SYNTHETIC dataset with:")
    print(f"  CSV path: {args.syn_csv_path}")
    print(f"  Image folder: {args.syn_img_folder}")
    print(f"  Score type: {args.score_name}")
    print(f"  Prompting: {args.prompting}")
    print(f"  Strategy: {args.strategy}")
    print(f"  N probes: {args.n_probes}")
    
    # Initialize scoring model
    print(f"\nInitializing {args.score_name} scoring model...")
    score_model = TextImageRepresentationalSimilarity(
        score_name=args.score_name,
        device=args.device
    )
    
    # Load synthetic dataset from CSV
    print(f"\nLoading synthetic dataset from {args.syn_csv_path}...")
    try:
        df = pd.read_csv(args.syn_csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return
    
    if df.empty:
        print("CSV is empty!")
        return
    
    print(f"Loaded {len(df)} synthetic samples from CSV")
    
    # to count the number of attributes
    dataset_gt = get_dataset(args.dataset, "train", getattr(args, "path_dataset", None), getattr(args, "path_gt", None), getattr(args, "path_gt_img", None))

    # Create a simple wrapper for synthetic data
    class SyntheticDatasetWrapper:
        """Wrapper for synthetic dataset from CSV and images folder."""
        def __init__(self, df, img_folder, dataset_name):
            self.df = df
            self.img_folder = img_folder
            self.dataset_name = dataset_name
            
            # Detect file extension from folder
            self.file_ext = None
            if os.path.exists(img_folder):
                for filename in os.listdir(img_folder):
                    if os.path.isfile(os.path.join(img_folder, filename)):
                        _, ext = os.path.splitext(filename)
                        if ext:
                            self.file_ext = ext
                            break
            
            if self.file_ext is None:
                self.file_ext = ".jpg"  # Default fallback
            
            # Get attribute names from dataset to align vectors
            src_path = os.path.join(os.path.dirname(__file__), "..", "..")
            if src_path not in sys.path:
                sys.path.insert(0, src_path)
            
            try:
                tmp_ds = get_dataset(dataset_name, "train", getattr(args, "path_dataset", None), getattr(args, "path_gt", None), getattr(args, "path_gt_img", None))
                self.listAllAttrib = tmp_ds.listAllAttrib if tmp_ds else []
            except Exception as e:
                print(f"Warning: Could not load attribute names from dataset: {e}")
                self.listAllAttrib = []
        
        def __len__(self):
            return len(self.df)
        
        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            
            img_filename = f"img-{idx}{self.file_ext}"
            img_path = os.path.join(self.img_folder, img_filename)
            
            if not os.path.exists(img_path):
                print(f"Warning: Image not found at {img_path}")
                return None
            
            try:
                img = Image.open(img_path).convert('RGB')
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                return None
            
            # Extract GT and selected vectors from *_gt / *_sel columns
            gt_vec = []
            sel_vec = []
            if self.listAllAttrib:
                for attr in self.listAllAttrib:
                    val_gt = row.get(f"{attr}_gt", 0)
                    val_sel = row.get(f"{attr}_sel", val_gt)
                    
                    try:
                        gt_vec.append(int(val_gt))
                    except Exception:
                        gt_vec.append(0)
                    
                    try:
                        sel_vec.append(int(val_sel))
                    except Exception:
                        sel_vec.append(0)
            else:
                # Fallback: use all numeric cols (excluding img) as both vectors
                arr = row.drop('img', errors='ignore').values
                gt_vec = arr.astype(int, copy=False)
                sel_vec = gt_vec

            return img, img_filename, np.array(gt_vec), np.array(sel_vec)
    
    syn_dataset = SyntheticDatasetWrapper(df, args.syn_img_folder, args.dataset)
    prompting_syn = create_prompting(syn_dataset, args.prompting, args.dataset, args.strategy)
    base_dir = args.lora_dir or os.path.dirname(os.path.abspath(args.syn_csv_path))
    experiment_path = _build_experiment_path(base_dir, args.dataset, args.prompting, args.score_name, args.strategy)
    
    # Compute scores for synthetic data
    print(f"\nComputing scores for synthetic samples...")
    imgpaths = []
    pos_scores = []
    neg_scores = []
    prompt_pos = []
    prompt_neg = []
    num_att_pos = []
    num_att_neg = []
    vector_pos = []
    vector_neg = []
    
    for idx in tqdm(range(len(syn_dataset))):
        sample = syn_dataset[idx]
        
        if sample is None:
            continue
        
        img, img_filename, gt_vector, selected_vector = sample
        
        if img is None or len(gt_vector) == 0:
            continue
        
        imgpaths.append(img_filename)
        
        # Generate positive (same as GT) and negative (flipped) target vectors
        target_pos = selected_vector.copy()
        
        # Generate probe prompts for positive case
        probes_pos, probes_neg = prompting_syn.generate_probes(
            gt_vector=gt_vector,
            target_vector=target_pos,
            n_probes=args.n_probes,
            strategy=args.strategy
        )
        
        if probes_pos is None or len(probes_pos) == 0:
            print(f"Warning: No positive probes generated for sample {idx}, skipping.")
            continue

        # Extract target_neg vector from first negative probe
        target_neg = probes_neg[0]['vector']
        

        # Compute scores for positive probes
        prompts_pos_list = [p['prompt'] for p in probes_pos]
        result_pos = score_model.score_prompts_images(prompts_pos_list, img)
        scores_pos = result_pos['scores']
        
        # Compute scores for negative probes
        prompts_neg_list = [p['prompt'] for p in probes_neg]
        result_neg = score_model.score_prompts_images(prompts_neg_list, img)
        scores_neg = result_neg['scores']
        
        # Store results
        pos_scores.append(scores_pos.tolist())
        neg_scores.append(scores_neg.tolist())
        prompt_pos.append(prompts_pos_list)
        prompt_neg.append(prompts_neg_list)
        num_att_pos.append(dataset_gt.getNumberOfAttributesFromVector(target_pos))
        num_att_neg.append(dataset_gt.getNumberOfAttributesFromVector(target_neg))
        vector_pos.append(target_pos.tolist())
        vector_neg.append(target_neg.tolist())
    
    save_scores_xlsx(
        syn_dataset, pos_scores, neg_scores, imgpaths, experiment_path, train=True,
        prompt_pos=prompt_pos, prompt_neg=prompt_neg,
        num_attr_pos=num_att_pos, num_attr_neg=num_att_neg,
        vector_pos=vector_pos, vector_neg=vector_neg,
        is_syn=True,
    )
    
    print(f"\n✓ Synthetic scoring complete! Results saved to: {experiment_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage B: Scoring and sample selection")
    parser.add_argument("--config", default=None, help="Path to config file")
    parser.add_argument("--getScores", action="store_true", help="Compute scores for real dataset")
    parser.add_argument("--getScoresSyn", action="store_true", help="Compute scores for synthetic dataset")
    
    # Arguments for score computation
    parser.add_argument("--dataset", type=str, default="RAPv2",
                        choices=["PA100k", "PETA", "PETAzs", "RAPv1", "RAPv2", "RAPzs"],
                        help="Dataset name")
    parser.add_argument("--score_name", type=str, default="blip",
                        choices=["blip"],
                        help="Type of scoring model to use")
    parser.add_argument("--prompting", type=str, default="fixed-rule",
                        help="Prompting strategy type (e.g., fixed-rule)")
    parser.add_argument("--strategy", type=str, default="identity",
                        help="Specific strategy (e.g.,identity)")
    parser.add_argument("--n_probes", type=int, default=1,
                        help="Number of probe prompts to generate")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (cuda/cpu, auto-detected if None)")
    parser.add_argument("--lora_dir", type=str, default=None,
                        help="Base directory (LoRA run folder) where Stage B outputs are stored")
    
    # Dataset custom paths
    parser.add_argument("--path_dataset", type=str, default=None,
                        help="Custom path to dataset images (overrides default paths)")
    parser.add_argument("--path_gt", type=str, default=None,
                        help="Custom path to ground truth pickle file")
    parser.add_argument("--path_gt_img", type=str, default=None,
                        help="Custom path to ground truth images folder")
    
    # Arguments for synthetic scoring
    parser.add_argument("--syn_csv_path", type=str, default="generated.csv",
                        help="Path to CSV file with synthetic data metadata")
    parser.add_argument("--syn_img_folder", type=str, default="genImg/",
                        help="Folder path containing synthetic images")
    
    args = parser.parse_args()
    
    # Load config from YAML if provided
    if args.config:
        if yaml is None:
            raise ImportError("PyYAML is required to load config files. Install with: pip install pyyaml")
        
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
        
        # Apply config values to args - CLI args take precedence if explicitly provided
        if not args.getScores and not args.getScoresSyn:
            args.getScores = config.get("getScores", False)
            args.getScoresSyn = config.get("getScoresSyn", False)
        
        if args.dataset == "RAPv2":
            args.dataset = config.get("dataset", "RAPv2")
        
        if args.score_name == "blip":
            args.score_name = config.get("score_name", "blip")
        
        if args.prompting == "fixed-rule":
            args.prompting = config.get("prompting", "fixed-rule")
        
        if args.strategy == "identity":
            args.strategy = config.get("strategy", "identity")
        
        if args.n_probes == 1:
            args.n_probes = config.get("n_probes", 1)
        
        if args.device is None:
            args.device = config.get("device")
        
        if args.lora_dir is None:
            args.lora_dir = config.get("lora_dir")
        
        if args.syn_csv_path == "generated.csv":
            args.syn_csv_path = config.get("syn_csv_path", "generated.csv")
        
        if args.syn_img_folder == "genImg/":
            args.syn_img_folder = config.get("syn_img_folder", "genImg/")
        
        # Dataset custom paths
        if args.path_dataset is None:
            args.path_dataset = config.get("path_dataset")
        
        if args.path_gt is None:
            args.path_gt = config.get("path_gt")
        
        if args.path_gt_img is None:
            args.path_gt_img = config.get("path_gt_img")
    
    if args.getScores:
        run_get_scores_real(args)
    elif args.getScoresSyn:
        run_get_scores_syn(args)
    else:
        if args.config:
            run_stage_b(args.config)
        else:
            print("Error: Must specify --getScores, --getScoresSyn, or --config")
