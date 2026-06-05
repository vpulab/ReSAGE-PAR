"""Stage C: Train/Test SI-Classifier on Stage B scores + label synthetic.

Modes:
  - train: uses real train scores from Stage B to fit a 1D threshold classifier.
  - test:  uses real test scores from Stage B to evaluate and save predictions.
  - labelingSyn: uses trained classifier and synthetic Stage B scores to output decisions
                 and pseudo-label vectors per sample.
"""

import argparse
import os
import pickle
import pandas as pd
import numpy as np

try:
    import yaml
except ImportError:
    yaml = None


from .pedestrian_metrics import get_pedestrian_metrics
from typing import Sequence, Optional, Dict, Any, List, Tuple
from collections import Counter, defaultdict

from .si_classifier import ThresholdClassifier, MetricBayesClassifier
from .labeling_policy import LabelingPolicy


# Strategy to column name mappings for training/testing
# Each entry can define which columns form the positive vector and (optionally) negative vector.
STRATEGY_COLUMN_MAP = {
    "identity": {"pos": "pos", "neg": "neg", "vector_pos_cols": ["pos"], "vector_neg_cols": ["neg"]},
    "pos": {"pos": "pos", "vector_pos_cols": ["pos"], "vector_neg_cols": []},
    # Placeholder for future strategies:
    # "interpolate:3": {"score": "interpolate_score", "vector_pos_cols": ["interpolate_score"], "vector_neg_cols": []},
    # "topk:5": {"pos": "topk_pos", "k": "topk_k", "vector_pos_cols": ["topk_pos"], "vector_neg_cols": []},
}


def _build_vectors(strategy: str, df: pd.DataFrame) -> tuple[list[list[float]], list[list[float]]]:
    """Return (vector, vector_neg) lists-of-lists based on strategy column mapping.

    - vector: list of per-row lists using mapped vector_pos_cols
    - vector_neg: list of per-row lists using mapped vector_neg_cols (may be empty lists)
    """
    if strategy not in STRATEGY_COLUMN_MAP:
        raise ValueError(f"Unknown strategy '{strategy}'. Supported: {list(STRATEGY_COLUMN_MAP.keys())}")

    col_map = STRATEGY_COLUMN_MAP[strategy]
    pos_cols = col_map.get("vector_pos_cols", [])
    neg_cols = col_map.get("vector_neg_cols", [])

    def _rows_for(cols: list[str]) -> list[list[float]]:
        rows: list[list[float]] = []
        for _, row in df.iterrows():
            vals = []
            for c in cols:
                if c in df.columns:
                    try:
                        vals.append(float(row[c]))
                    except Exception:
                        vals.append(0.0)
                else:
                    vals.append(0.0)
            rows.append(vals)
        return rows

    return _rows_for(pos_cols), _rows_for(neg_cols)


def _exp_paths(dataset: str, prompting: str, score_name: str, clf_tag: str, strategy: str = "identity", base_dir: str | None = None) -> Tuple[str, str, str]:
    root = os.path.abspath(base_dir) if base_dir else os.path.abspath(".")
    base_real = os.path.join(root, f"{dataset}_{prompting}_{score_name}_{strategy}_scores")
    # Stage B writes synthetic scores into the same folder, not a separate _syn folder
    base_syn = base_real
    artifacts = os.path.join(root, f"{dataset}_{prompting}_{score_name}_{strategy}_{clf_tag}_si", "artifacts")
    return base_real, base_syn, artifacts


def _load_scores_csv(path: str, train: bool = True) -> pd.DataFrame:
    """Load scores from CSV file. Supports both train and test modes."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing scores CSV at {path}")
    df = pd.read_csv(path)
    return df


def _load_scores_xlsx(path: str, train: bool) -> pd.DataFrame:
    """Load the 'prompting' sheet from XLSX file (default view used across code)."""
    sheet = "prompting"
    df = pd.read_excel(path, sheet_name=sheet, engine="openpyxl")
    return df

def _load_scores_xlsx_sheet(path: str, sheet: str) -> pd.DataFrame:
    """Load a specific sheet from an XLSX file."""
    return pd.read_excel(path, sheet_name=sheet, engine="openpyxl")


def _parse_vector_cell(cell) -> list[float]:
    """Parse a cell that may contain a list-like repr or scalar into list[float]."""
    if isinstance(cell, list):
        return [float(x) for x in cell]
    if isinstance(cell, (np.ndarray, tuple)):
        return [float(x) for x in cell.tolist()]
    if isinstance(cell, str):
        txt = cell.strip()
        # basic bracketed list detection
        if (txt.startswith("[") and txt.endswith("]")) or (txt.startswith("(") and txt.endswith(")")):
            try:
                import json
                return [float(x) for x in json.loads(txt.replace("(", "[").replace(")", "]"))]
            except Exception:
                pass
        try:
            return [float(txt)]
        except Exception:
            return []
    try:
        return [float(cell)]
    except Exception:
        return []




def _make_classifier(args):
    if getattr(args, "classifier", None) in (None, "threshold"):
        return ThresholdClassifier(), "threshold"
    if args.classifier == "bayes":
        mode = getattr(args, "bayes_mode", "gauss")
        prior = getattr(args, "prior_pos", None)
        return MetricBayesClassifier(mode=mode, prior_pos=prior), f"bayes-{mode}"
    raise ValueError(f"Unknown classifier: {getattr(args, 'classifier', None)}")


def _save_clf_tag(artifacts: str, clf_tag: str):
    os.makedirs(artifacts, exist_ok=True)
    with open(os.path.join(artifacts, "classifier_tag.txt"), "w") as f:
        f.write(clf_tag)


def _load_classifier_generic(clf_path: str):
    with open(clf_path, "rb") as f:
        obj = pickle.load(f)
    # direct class instances
    if isinstance(obj, ThresholdClassifier) or isinstance(obj, MetricBayesClassifier):
        return obj
    # legacy dict for ThresholdClassifier
    if isinstance(obj, dict) and "threshold" in obj:
        return ThresholdClassifier(threshold=float(obj.get("threshold", 0.0)))
    raise TypeError(f"Unsupported classifier pickle at {clf_path}: {type(obj)}")


def _split_train_val(arr: np.ndarray, val_frac: float, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]: 
    
    n = arr.size 
    if n == 0: 
        return arr, arr
    
    idx = rng.permutation(n) 
    n_val = int(np.ceil(val_frac * n)) 
    n_val = min(max(n_val, 1 if n > 1 else 0), n - 1 if n > 1 else 0) 
    # keep at least 1 in train if possible 
    val = arr[idx[:n_val]] 
    train = arr[idx[n_val:]] 
    return train, val


def _as_1d_float(a: Any) -> np.ndarray:
    x = np.asarray(a, dtype=float).ravel()
    return x[np.isfinite(x)]


def _predict_labels(clf, x: np.ndarray) -> np.ndarray:
    """Return 0/1 labels for x using whatever interface the classifier exposes."""
    # 1) predict
    if hasattr(clf, "predict"):
        y = np.asarray(clf.predict(x)).ravel()
        if y.dtype.kind in "fc":  # continuous -> threshold
            if np.nanmin(y) >= 0 and np.nanmax(y) <= 1:
                return (y >= 0.5).astype(int)
            return (y >= 0.0).astype(int)
        return y.astype(int)

    # 2) predict_proba (Nx2 or N)
    for name in ("predict_proba", "proba", "posterior", "posterior_proba"):
        if hasattr(clf, name):
            p = np.asarray(getattr(clf, name)(x))
            if p.ndim == 2:
                # assume column for positive is the max-proba column
                pos = np.argmax(np.mean(p, axis=0))
                return (p[:, pos] >= 0.5).astype(int)
            if p.ndim == 1:
                return (p >= 0.5).astype(int)

    # 3) decision-like scores
    for name in ("decision_function", "log_likelihood_ratio", "score_samples"):
        if hasattr(clf, name):
            s = np.asarray(getattr(clf, name)(x)).ravel()
            return (s >= 0.0).astype(int)

    raise AttributeError("Classifier does not expose a usable prediction method.")

def _split_train_val_idx(n: int, val_frac: float, rng: np.random.Generator):
    if n == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    idx = rng.permutation(n)
    n_val = int(np.ceil(val_frac * n))
    n_val = min(max(n_val, 1 if n > 1 else 0), n - 1 if n > 1 else 0)

    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    return train_idx, val_idx

listColumnsNotAttributes=["pos/neg", "imgpath", "score", "decision"]

def executeTrainIdentityStrategy(strategy,
                               df: pd.DataFrame,
                               clf,
                               clf_tag: str,
                               artifacts: str,
                               imgpaths: list[str],
                               attr_names: list[str],
                               activated_vectors: list[list[int]],
                               prob_threshold: float,
                               random_state: int = 0,
                               val_frac: float = 0.20,
                               labeling_policy: LabelingPolicy = None,
                               args=None):
    global listColumnsNotAttributes
    col_map = STRATEGY_COLUMN_MAP[strategy]
    pos_col = col_map.get("pos", "pos")
    neg_col = col_map.get("neg")
    
    # Load columns based on strategy
    pos_vecs = [_parse_vector_cell(v) for v in df[pos_col]] if pos_col in df.columns else None
    neg_vecs = [_parse_vector_cell(v) for v in df[neg_col]] if neg_col in df.columns else None
    pos = np.array([v[0] if v else 0.0 for v in pos_vecs], dtype=float) if pos_vecs is not None else None
    neg = np.array([v[0] if v else 0.0 for v in neg_vecs], dtype=float) if neg_vecs is not None else None
    
    n_pos, n_neg = pos.size, neg.size

    n = len(df)
    rng = np.random.default_rng(random_state)


    train_idx, val_idx = _split_train_val_idx(n, val_frac, rng)
    pos_tr, pos_val = pos[train_idx], pos[val_idx]
    neg_tr, neg_val = neg[train_idx], neg[val_idx]

    if pos is None or (neg is None and neg_col):
        raise ValueError(f"scores_train CSV/XLSX must contain '{pos_col}' and '{neg_col}' columns for strategy '{strategy}'")

    # Fit classifier; for Bayes KDE allow configurable bandwidth
    if hasattr(clf, "mode") and getattr(clf, "mode") == "kde":
        bandwidth = None
        if args is not None and hasattr(args, "kde_bandwidth") and getattr(args, "kde_bandwidth") not in (None, ""):
            bandwidth = getattr(args, "kde_bandwidth")
        clf.fit(pos_tr, neg_tr, bandwidth=(bandwidth if bandwidth is not None else "scott"))
    else:
        clf.fit(pos_tr, neg_tr)
    clf_path = os.path.join(artifacts, "classifier.pkl")
    clf.save(clf_path)

    _save_clf_tag(artifacts, clf_tag)
    msg = f"[train] Trained {clf_tag} with strategy '{strategy}'. Saved to {clf_path}"
    print(msg)

    X_val = np.concatenate([pos_val, neg_val])
    y_val = np.concatenate([np.ones(pos_val.size, int), np.zeros(neg_val.size, int)])

    # Many simple Bayes classifiers take a 1D array of scores
    y_hat = _predict_labels(clf, X_val)

    val_acc = float(np.mean(y_hat == y_val))
    pos_acc = float(np.mean(y_hat[:pos_val.size] == 1)) if pos_val.size else float("nan")
    neg_acc = float(np.mean(y_hat[pos_val.size:] == 0)) if neg_val.size else float("nan")

    # basic stats
    stats = {
        "n_pos": float(n_pos),
        "n_neg": float(n_neg),
        "n_pos_train": float(pos_tr.size),
        "n_neg_train": float(neg_tr.size),
        "n_pos_val": float(pos_val.size),
        "n_neg_val": float(neg_val.size),
        "mu_pos_train": float(np.mean(pos_tr)) if pos_tr.size else float("nan"),
        "mu_neg_train": float(np.mean(neg_tr)) if neg_tr.size else float("nan"),
        "val_acc": val_acc,
        "val_pos_acc": pos_acc,
        "val_neg_acc": neg_acc,
    }

    preds_pos_train = clf.predict(pos_tr, thresh=prob_threshold)
    labels, decisions = labeling_policy.decide_labels(preds_pos_train, prob_threshold, activated_vectors)
    rows = []
    for i in range(len(labels)):
        rows.append(["pos", imgpaths[i], pos_tr[i], decisions[i]] + labels[i])
    
    preds_neg_train = clf.predict(neg_tr, thresh=prob_threshold)
    labels, decisions = labeling_policy.decide_labels(preds_neg_train, prob_threshold, activated_vectors)

    rows = []
    for i in range(len(labels)):
        rows.append(["neg", imgpaths[i], neg_tr[i], decisions[i]] + labels[i])    

    
    out_df = pd.DataFrame(rows, columns=listColumnsNotAttributes + attr_names)
    out_csv = os.path.join(artifacts, "train_predictions.csv")
    out_df.to_csv(out_csv, index=False)
    print(f"[train] Saved predicted vectors to {out_csv}")


    gettingPedestrianMetrics(out_df, artifacts, "train")

    preds_pos_val = clf.predict(pos_val, thresh=prob_threshold)
    labels, decisions = labeling_policy.decide_labels(preds_pos_val, prob_threshold, activated_vectors)
    rows = []
    for i in range(len(labels)):
        rows.append(["pos", imgpaths[i], pos_val[i], decisions[i]] + labels[i])
    
    preds_neg_val = clf.predict(neg_val, thresh=prob_threshold)
    labels, decisions = labeling_policy.decide_labels(preds_neg_val, prob_threshold, activated_vectors)

    rows = []
    for i in range(len(labels)):
        rows.append(["neg", imgpaths[i], neg_val[i], decisions[i]] + labels[i])   


    out_df = pd.DataFrame(rows, columns=listColumnsNotAttributes + attr_names)
    out_csv = os.path.join(artifacts, "val_predictions.csv")
    out_df.to_csv(out_csv, index=False)
    print(f"[val] Saved predicted vectors to {out_csv}")

    gettingPedestrianMetrics(out_df, artifacts, "val")
    return stats




def cmd_train(args):

    clf, clf_tag = _make_classifier(args)
    strategy = getattr(args, "strategy", "identity")
    base_real, _, artifacts = _exp_paths(args.dataset, args.prompting, args.score_name, clf_tag, strategy, getattr(args, "lora_dir", None))
    os.makedirs(artifacts, exist_ok=True)
    
    # Try CSV first, fall back to XLSX
    train_xlsx = os.path.join(base_real, "scores_train.xlsx")
    
    df_sanity = None

    if os.path.isfile(train_xlsx):
        df = _load_scores_xlsx(train_xlsx, train=True)
        df_sanity = _load_scores_xlsx_sheet(train_xlsx, "sanity_check")
    else:
        raise FileNotFoundError(f"Missing real train scores at or {train_xlsx}")
    
    # Get column names for this strategy
    if strategy not in STRATEGY_COLUMN_MAP:
        raise ValueError(f"Unknown strategy '{strategy}'. Supported: {list(STRATEGY_COLUMN_MAP.keys())}")
    
    # Build activated vectors from sanity_check
    if "imgpath" in df.columns:
        imgpaths = [os.path.basename(p) for p in df["imgpath"].astype(str).tolist()]
    else:
        imgpaths = [f"row_{i}" for i in range(len(df))]
    attr_names: list[str] = []
    activated_vectors: list[list[int]] = []
    if df_sanity is not None:
        raw_pos_cols = [c for c in df_sanity.columns if str(c).endswith("_pos")]
        # Exclude non-attribute columns present in sanity_check
        exclude_bases = {"prompt", "num_attr"}
        pos_cols = [c for c in raw_pos_cols if c[:-4] not in exclude_bases]
        attr_names = [c[:-4] for c in pos_cols]
        mat = df_sanity[pos_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
        mat = np.clip(np.rint(mat), 0, 1).astype(int)
        mat = np.where(mat == 0, -1, 1)
        activated_vectors = [row.tolist() for row in mat]
    else:
        raise ValueError(f"Df sanity is None for training data at {train_xlsx}")


    prob_threshold = float(getattr(args, "threshold", 0.5))
    
    labeling_policy_type = getattr(args, "labeling_policy", "hardlabeling")
    labeling_policy = LabelingPolicy(policy=labeling_policy_type)

    if strategy == "identity":
        stats = executeTrainIdentityStrategy(
            strategy, df, clf, clf_tag, artifacts, imgpaths, attr_names, activated_vectors,
            prob_threshold, random_state=args.random_state, val_frac=args.val_frac,
            labeling_policy=labeling_policy, args=args
        )
    else:
        raise NotImplementedError(f"Training for strategy '{strategy}' not implemented yet.")
    
    train_txt = os.path.join(artifacts, "results_train.txt")
    _save_text(train_txt, json.dumps(stats, indent=2))
    print(f"Metrics saved to: {train_txt}")

    return stats

def executeTestIdentityStrategy(strategy,
                              df: pd.DataFrame,
                              clf,
                              clf_tag: str,
                              artifacts: str,
                              imgpaths: list[str],
                              attr_names: list[str],
                              activated_vectors: list[list[int]],
                              prob_threshold: float,
                              labeling_policy: LabelingPolicy = None):
    
    global listColumnsNotAttributes
    col_map = STRATEGY_COLUMN_MAP[strategy]
    pos_col = col_map.get("pos", "pos")
    neg_col = col_map.get("neg")


    pos_vecs = [_parse_vector_cell(v) for v in df[pos_col]] if pos_col in df.columns else None
    neg_vecs = [_parse_vector_cell(v) for v in df[neg_col]] if neg_col in df.columns else None
    if pos_vecs is None:
        raise ValueError(f"scores_test CSV/XLSX must contain '{pos_col}' column for strategy '{strategy}'")

    pos_scores = np.array([v[0] if v else 0.0 for v in pos_vecs], dtype=float)
    neg_scores = np.array([v[0] if v else 0.0 for v in neg_vecs], dtype=float) if neg_vecs is not None else None
    
    

    n_pos, n_neg = pos_scores.size, neg_scores.size

    # --- Validate on the held-out 20% ---
    X_val = np.concatenate([pos_scores, neg_scores])
    y_val = np.concatenate([np.ones(pos_scores.size, int), np.zeros(neg_scores.size, int)])

    # Many simple Bayes classifiers take a 1D array of scores
    y_hat = _predict_labels(clf, X_val)

    test_acc = float(np.mean(y_hat == y_val))
    test_pos_acc = float(np.mean(y_hat[:pos_scores.size] == 1)) if pos_scores.size else float("nan")
    test_neg_acc = float(np.mean(y_hat[pos_scores.size:] == 0)) if neg_scores.size else float("nan")

    # basic stats
    stats = {
        "n_pos_test": float(n_pos),
        "n_neg_test": float(n_neg),
        "mu_pos_test": float(np.mean(pos_scores)) if pos_scores.size else float("nan"),
        "mu_neg_test": float(np.mean(neg_scores)) if neg_scores.size else float("nan"),
        "test_acc": test_acc,
        "test_pos_acc": test_pos_acc,
        "test_neg_acc": test_neg_acc,
    }
    # --- Save predictions on test set ---
    preds_pos = clf.predict(pos_scores, thresh=prob_threshold)
    labels, decisions = labeling_policy.decide_labels(preds_pos, prob_threshold, activated_vectors)
    rows = []
    for i in range(len(labels)):
        rows.append(["pos", imgpaths[i], pos_scores[i], decisions[i]] + labels[i])

    
    preds_neg = clf.predict(neg_scores, thresh=prob_threshold)
    labels, decisions = labeling_policy.decide_labels(preds_neg, prob_threshold, activated_vectors)

    rows = []
    for i in range(len(labels)):
        rows.append(["neg", imgpaths[i], neg_scores[i], decisions[i]] + labels[i])


    out_df = pd.DataFrame(rows, columns=listColumnsNotAttributes + attr_names)
    out_csv = os.path.join(artifacts, "test_predictions.csv")
    out_df.to_csv(out_csv, index=False)
    print(f"[test] Saved predictions to {out_csv}")

    gettingPedestrianMetrics(out_df, artifacts, "test")


    return stats

def cmd_test(args):
    _, clf_tag = _make_classifier(args)
    strategy = getattr(args, "strategy", "identity")
    base_real, _, artifacts = _exp_paths(args.dataset, args.prompting, args.score_name, clf_tag, strategy, getattr(args, "lora_dir", None))
    
    # Try CSV first, fall back to XLSX
    test_xlsx = os.path.join(base_real, "scores_test.xlsx")
    
    df_sanity = None

    if os.path.isfile(test_xlsx):
        df = _load_scores_xlsx(test_xlsx, train=False)
        df_sanity = _load_scores_xlsx_sheet(test_xlsx, "sanity_check")
    else:
        raise FileNotFoundError(f"Missing real test scores at {test_xlsx}")
    
    clf_path = os.path.join(artifacts, "classifier.pkl")
    if not os.path.isfile(clf_path):
        raise FileNotFoundError(f"Missing classifier at {clf_path}. Run train first.")

    clf = _load_classifier_generic(clf_path)
    
    # Get column names for this strategy
    if strategy not in STRATEGY_COLUMN_MAP:
        raise ValueError(f"Unknown strategy '{strategy}'. Supported: {list(STRATEGY_COLUMN_MAP.keys())}")
    
        
    # Build activated vectors from sanity_check
    if "imgpath" in df.columns:
        imgpaths = [os.path.basename(p) for p in df["imgpath"].astype(str).tolist()]
    else:
        imgpaths = [f"row_{i}" for i in range(len(df))]
    attr_names: list[str] = []
    activated_vectors: list[list[int]] = []
    if df_sanity is not None:
        raw_pos_cols = [c for c in df_sanity.columns if str(c).endswith("_pos")]
        exclude_bases = {"prompt", "num_attr"}
        pos_cols = [c for c in raw_pos_cols if c[:-4] not in exclude_bases]
        attr_names = [c[:-4] for c in pos_cols]
        mat = df_sanity[pos_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
        mat = np.clip(np.rint(mat), 0, 1).astype(int)
        mat = np.where(mat == 0, -1, 1)
        activated_vectors = [row.tolist() for row in mat]
    else:
        raise ValueError(f"df_sanity is None for test data at {test_xlsx}")
    
    prob_threshold = float(getattr(args, "threshold", 0.5))
    
    labeling_policy_type = getattr(args, "labeling_policy", "hardlabeling")
    labeling_policy = LabelingPolicy(policy=labeling_policy_type)

    if strategy == "identity":
        stats = executeTestIdentityStrategy(strategy, df, clf, clf_tag, artifacts, imgpaths, attr_names, activated_vectors, prob_threshold, labeling_policy)
    else:
        raise NotImplementedError(f"Training for strategy '{strategy}' not implemented yet.")
   
    test_txt = os.path.join(artifacts, "results_test.txt")
    _save_text(test_txt, json.dumps(stats, indent=2))
    print(f"Metrics saved to: {test_txt}")

    return

import json
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):   return obj.tolist()
        if isinstance(obj, np.integer):   return int(obj)
        if isinstance(obj, np.floating):  return float(obj)
        if isinstance(obj, np.bool_):     return bool(obj)
        return super().default(obj)

def gettingPedestrianMetrics(df: pd.DataFrame, artifacts: str = ".", type=""):
    """
    Process pedestrian metrics from a dataframe with imgpath and vector columns.
    
    For each image and its corresponding vector, calls get_ground_truth() to retrieve
    ground truth paths and labels, then processes the results.
    
    Args:
        df: DataFrame with columns ["imgpath", ...vector_columns]
            - imgpath: path to the image
            - remaining columns: vector values
    
    Returns:
        List of tuples (imgpath, vector, gt_paths, gt_labels) for each row
    """
    global listColumnsNotAttributes
    if df.empty:
        print(" Warning: empty dataframe")
        return []
    
    if "imgpath" not in df.columns:
        raise ValueError("DataFrame must contain 'imgpath' column")
    
    # Identify vector columns (all columns except imgpath)
    vector_cols = [c for c in df.columns if c not in listColumnsNotAttributes]
    
    if not vector_cols:
        print("Warning: no vector columns found in dataframe")
        return []
    
    results = []
    all_img_paths = []
    all_vectors= []
    for idx, row in df.iterrows():
        imgpath = str(row["imgpath"])
        # Extract vector from remaining columns
        vector = [float(row[c]) if pd.notna(row[c]) else 0.0 for c in vector_cols]
        all_img_paths.append(imgpath)
        all_vectors.append(vector)

        
    # Call placeholder function to get ground truth
    gt_paths, gt_labels, listAttributes = get_ground_truth()

    out = accuracy_report_from_paths(gt_paths, gt_labels, all_img_paths, all_vectors, listAttributes)

    pedestrianDict = out["pedestrian"]

    out_json = os.path.join(artifacts, "pseudolabels_syn"+type+".json")                

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(pedestrianDict, f, cls=NumpyEncoder, indent=2, ensure_ascii=False, sort_keys=True)
   
    print("Pedestrian metrics and pseudolabels saved to:", out_json)

    return results




def accuracy_report(
    y_true: Sequence, 
    y_pred: Sequence, 
    attributes: Optional[Sequence[str]] = None
) -> Dict[str, Any]:
    """
    Compute accuracy with support for multi-attribute (multi-column) labels.

    Counting rule (as requested):
    - per-attribute "count": number of predictions made (y_pred in {0,1}); i.e., excludes -1 predictions.
    - Accuracy uses only positions where BOTH y_true in {0,1} and y_pred in {0,1}.
    - y_true == -1 (or not in {0,1}) is ignored for accuracy.

    Args:
        y_true: shape (N,) or (N, K). Values in {0,1,-1}; -1 means "not applicable".
        y_pred: shape (N,) or (N, K). Values typically {0,1,-1}; -1 means "no prediction".
        attributes: optional list of K attribute names; if None, uses ["attr_0", ..., "attr_{K-1}"].

    Returns:
        {
          "overall_accuracy": float,                 # micro accuracy over entries used for accuracy
          "overall_micro_accuracy": float,           # alias
          "overall_macro_accuracy": float,           # mean of per-attribute accuracies (attrs with count_for_accuracy>0)
          "total_predictions_count": int,            # sum over attrs of y_pred in {0,1}
          "total_count_for_accuracy": int,           # sum over attrs where both y_true,y_pred in {0,1}
          "per_attribute": {
              "<attr_name>": {
                  "accuracy": float,                 # NaN if count_for_accuracy==0
                  "count": int,                      # predictions made (y_pred in {0,1})
                  "count_for_accuracy": int          # used in accuracy (both in {0,1})
              },
              ...
          }
        }
    """
    y = np.asarray(y_true)
    h = np.asarray(y_pred)

    # Ensure 2D
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    if h.ndim == 1:
        h = h.reshape(-1, 1)

    if y.shape != h.shape:
        raise ValueError(f"Shape mismatch: y_true {y.shape} vs y_pred {h.shape}")

    N, K = y.shape
    if attributes is None:
        attributes = [f"attr_{j}" for j in range(K)]
    else:
        if len(attributes) != K:
            raise ValueError(f"`attributes` length ({len(attributes)}) must match number of columns K={K}")

    # Masks
    pred_valid   = np.isin(h, [0, 1])         # predictions made (NOT -1)
    gt_valid     = np.isin(y, [0, 1])         # ground truth available
    used_mask    = pred_valid & gt_valid      # actually contributes to accuracy

    # Per-attribute stats
    per_attr: Dict[str, Dict[str, float | int]] = {}
    per_attr_acc = []
    total_pred = int(np.sum(pred_valid))
    total_used = int(np.sum(used_mask))

    for j, name in enumerate(attributes):
        cnt_pred = int(np.sum(pred_valid[:, j]))
        cnt_used = int(np.sum(used_mask[:, j]))

        if cnt_used > 0:
            acc_j = float(np.mean(y[used_mask[:, j], j] == h[used_mask[:, j], j]))
            per_attr_acc.append(acc_j)
        else:
            acc_j = float("nan")

        per_attr[name] = {
            "accuracy": acc_j,
            "count": cnt_pred,                    # as requested: only depends on prediction != -1
            "count_for_accuracy": cnt_used,       # entries that actually contributed to accuracy
        }

    # Micro accuracy over all entries used for accuracy
    if total_used > 0:
        overall_micro = float(np.mean((y[used_mask]) == (h[used_mask])))
    else:
        overall_micro = 0.0

    # Macro accuracy: mean of per-attribute accuracies where count_for_accuracy>0
    if per_attr_acc:
        overall_macro = float(np.mean(per_attr_acc))
    else:
        overall_macro = 0.0

    return {
        "overall_accuracy": overall_micro,
        "overall_micro_accuracy": overall_micro,
        "overall_macro_accuracy": overall_macro,
        "total_predictions_count": total_pred,
        "total_count_for_accuracy": total_used,
        "per_attribute": per_attr,
    }


def zip_to_dict(keys: list[str], values: list[list]) -> dict[str, list]:
    """
    Asigna un dict donde cada string de `keys` se empareja con la sublista en `values`.
    Requiere misma longitud.
    """
    if len(keys) != len(values):
        raise ValueError(f"Longitudes distintas: keys={len(keys)} vs values={len(values)}")
    return dict(zip(keys, values))

def accuracy_report_from_paths(
    gt_paths, gt_labels, pred_img_paths, pred_vectors, listAttributes) -> Dict[str, Any]:
    """
    For each prediction row (pred_paths[i], pred_vectors[i]), find the GT vector for the same image
    and compare that pair. Duplicated pred_paths are all evaluated separately.

    Returns accuracy_report(y_true, y_pred, ...) + diagnostics.
    """
    
    gt = zip_to_dict(gt_paths, gt_labels)
    y_true = []
    y_pred = []
    for imgpath, pred in zip(pred_img_paths, pred_vectors):
        
        gt_vector = gt[imgpath]
        y_true.append(np.asarray(gt_vector).ravel())
        y_pred.append(np.asarray(pred).ravel())



    y = np.vstack(y_true)
    h = np.vstack(y_pred)

    # Your existing metric functions (assumed available)
    out = accuracy_report(y, h, attributes=listAttributes)
    ped = get_pedestrian_metrics(y, h)

    out.update({
        "matched": int(y.shape[0]),
        
        "pedestrian": ped,
    })
    return out



def get_ground_truth():
    global dataset
    ds = dataset

    imgpaths=[]
    vectors=[]
    for img in ds.all_images:
        vector=ds.getLabelVector(img)
        imgpaths.append(img)
        vectors.append(vector)

    return imgpaths, vectors, list(ds.dataPkl['attr_name'])

def executeSynLabelingIdentityStrategy(strategy,
                                     df_prom: pd.DataFrame,
                                     clf,
                                     artifacts: str,
                                     args,
                                     imgpaths: list[str],
                                     attr_names: list[str],
                                     activated_vectors: list[list[int]],
                                     prob_threshold: float,
                                     labeling_policy: LabelingPolicy = None):
    global listColumnsNotAttributes
    col_map = STRATEGY_COLUMN_MAP[strategy]
    pos_col = col_map.get("pos", "pos")
    
    if pos_col not in df_prom.columns:
        raise ValueError(f"Synthetic scores CSV/XLSX must contain '{pos_col}' column for strategy '{strategy}'")
    pos_vecs = [_parse_vector_cell(v) for v in df_prom[pos_col]]
    scores = np.array([v[0] if v else 0.0 for v in pos_vecs], dtype=float)
    probs = clf.predict_proba(scores)
    
    labels, decisions = labeling_policy.decide_labels(probs, prob_threshold, activated_vectors)

    rows = []
    for i in range(len(labels)):
        rows.append(["pos", imgpaths[i], scores[i], decisions[i]] + labels[i])

    cols = listColumnsNotAttributes + attr_names
    out_df = pd.DataFrame(rows, columns=cols)
    os.makedirs(artifacts, exist_ok=True)
    out_csv = os.path.join(artifacts, "pseudolabels_syn.csv")
    out_df.to_csv(out_csv, index=False)
    print(f"[labelingSyn] Saved synthetic pseudo-labels to {out_csv}")

    return

def cmd_labeling_syn(args):
    _, clf_tag = _make_classifier(args)
    strategy = getattr(args, "strategy", "identity")
    _, base_syn, artifacts = _exp_paths(args.dataset, args.prompting, args.score_name, clf_tag, strategy, getattr(args, "lora_dir", None))
    
    # Synthetic now consolidated as scores_syn.* in the same folder
    syn_scores_xlsx = os.path.join(base_syn, "scores_syn.xlsx")

    if os.path.isfile(syn_scores_xlsx):
        df_prom = _load_scores_xlsx(syn_scores_xlsx, train=True)
        # Load sanity_check sheet for activated attributes
        df_sanity = _load_scores_xlsx_sheet(syn_scores_xlsx, "sanity_check")
    else:
        raise FileNotFoundError(f"Missing synthetic scores at {syn_scores_xlsx}")
    
    clf_path = os.path.join(artifacts, "classifier.pkl")
    if not os.path.isfile(clf_path):
        raise FileNotFoundError(f"Missing classifier at {clf_path}. Run train first.")

    clf = _load_classifier_generic(clf_path)
    
    # Build activated attribute vectors from sanity_check sheet (preferred)
    attr_names: list[str] = []
    activated_vectors: list[list[int]] = []
    if df_sanity is not None:
        raw_pos_cols = [c for c in df_sanity.columns if str(c).endswith("_pos")]
        exclude_bases = {"prompt", "num_attr"}
        pos_cols = [c for c in raw_pos_cols if c[:-4] not in exclude_bases]
        attr_names = [c[:-4] for c in pos_cols]
        mat = df_sanity[pos_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
        mat = np.clip(np.rint(mat), 0, 1).astype(int)
        mat = np.where(mat == 0, -1, 1)
        activated_vectors = [row.tolist() for row in mat]
    else:
        print("[labelingSyn] Warning: no sanity_check sheet found; activated attribute vectors will be empty.")
        # Fallback: try to synthesize zero-length vectors (will yield all -1)
        activated_vectors = [[] for _ in range(len(df_prom))]
        attr_names = []

    #print(activated_vectors)

    if "imgpath" in df_prom.columns:
        imgpaths = df_prom["imgpath"].astype(str).tolist()
    else:
        imgpaths = [f"img_{i}" for i in range(len(df_prom))]

    # Get column names for this strategy
    if strategy not in STRATEGY_COLUMN_MAP:
        raise ValueError(f"Unknown strategy '{strategy}'. Supported: {list(STRATEGY_COLUMN_MAP.keys())}")
    

    labeling_policy_type = getattr(args, "labeling_policy", "hardlabeling")
    labeling_policy = LabelingPolicy(policy=labeling_policy_type)

    if strategy == "identity":
        executeSynLabelingIdentityStrategy(strategy, df_prom, clf, artifacts, args, imgpaths, attr_names, activated_vectors, prob_threshold=float(getattr(args, "threshold", 0.5)), labeling_policy=labeling_policy)
    else:
        raise NotImplementedError(f"Synthetic labeling for strategy '{strategy}' not implemented yet.")
    


def _save_text(path: str, text: str):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


import json

def get_dataset(dataset_name: str, split: str = "train", path_dataset: str | None = None, path_gt: str | None = None, path_gt_img: str | None = None):
    """Load dataset by name and split.
    
    Args:
        dataset_name: Name of dataset (PA100k, PETA, PETAzs, RAPv1, RAPv2, RAPzs)
        split: 'train' or 'test'
    
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

dataset=None
import sys
def main():
    global dataset
    ap = argparse.ArgumentParser(description="Stage C: SI-Classifier + Labeling Policy")
    ap.add_argument("--config", type=str, default=None, help="Path to YAML config file (takes precedence over CLI args)")
    ap.add_argument("mode", nargs="?", choices=["train", "test", "labelingSyn"], help="Stage C mode")
    ap.add_argument("--dataset", default="RAPzs")
    ap.add_argument("--prompting", default="fixed-rule")
    ap.add_argument("--score_name", default="blip")
    ap.add_argument("--strategy", default="identity", choices=list(STRATEGY_COLUMN_MAP.keys()), 
                    help="Score strategy (determines column names in CSV). Supported: " + ", ".join(STRATEGY_COLUMN_MAP.keys()))
    ap.add_argument("--threshold", type=float, default=0.5, help="Decision threshold for labelingSyn")
    ap.add_argument("--classifier", choices=["threshold", "bayes"], default="threshold", help="Select classifier")
    ap.add_argument("--bayes_mode", choices=["gauss", "kde"], default="gauss", help="Bayes classifier density model")
    ap.add_argument("--prior_pos", type=float, default=None, help="Bayes classifier prior for positive class (0-1)")
    ap.add_argument("--kde_bandwidth", type=str, default="scott", help="KDE bandwidth (float or 'scott')")
    ap.add_argument("--lora_dir", default=None, help="Base directory (LoRA run folder) for Stage B/C outputs")
    ap.add_argument("--labeling_policy", choices=["hardlabelling"], default="hardlabelling", help="Labeling policy for synthetic data")
    ap.add_argument("--random_state", type=int, default=42, help="Random state for reproducibility")
    ap.add_argument("--val_frac", type=float, default=0.2, help="Fraction of data to use for validation")
    ap.add_argument("--path_dataset", type=str, default=None, help="Custom path to dataset images")
    ap.add_argument("--path_gt", type=str, default=None, help="Custom path to ground truth pickle file")
    ap.add_argument("--path_gt_img", type=str, default=None, help="Custom path to ground truth images folder")

    args = ap.parse_args()
    
    # Load config from YAML if provided
    if args.config:
        if yaml is None:
            raise ImportError("PyYAML is required to load config files. Install with: pip install pyyaml")
        
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
        
        # Apply config values to args - CLI args take precedence if explicitly provided
        if args.mode is None:
            args.mode = config.get("mode")
        
        if args.dataset == "RAPzs":
            args.dataset = config.get("dataset", "RAPzs")
        
        # Optional dataset path overrides
        if not hasattr(args, "path_dataset") or args.path_dataset is None:
            args.path_dataset = config.get("path_dataset")
        if not hasattr(args, "path_gt") or args.path_gt is None:
            args.path_gt = config.get("path_gt")
        if not hasattr(args, "path_gt_img") or args.path_gt_img is None:
            args.path_gt_img = config.get("path_gt_img")
        
        if args.prompting == "fixed-rule":
            args.prompting = config.get("prompting", "fixed-rule")
        
        if args.score_name == "blip":
            args.score_name = config.get("score_name", "blip")
        
        if args.strategy == "identity":
            args.strategy = config.get("strategy", "identity")
        
        if args.threshold == 0.5:
            args.threshold = float(config.get("threshold", 0.5))
        
        if args.classifier == "threshold":
            args.classifier = config.get("classifier", "threshold")
        
        if args.bayes_mode == "gauss":
            args.bayes_mode = config.get("bayes_mode", "gauss")
        if args.prior_pos is None:
            args.prior_pos = config.get("prior_pos")
        if not hasattr(args, "kde_bandwidth") or args.kde_bandwidth == "scott":
            args.kde_bandwidth = config.get("kde_bandwidth", "scott")
        
        if args.lora_dir is None:
            args.lora_dir = config.get("lora_dir")
        
        if args.labeling_policy == "hardlabelling":
            args.labeling_policy = config.get("labeling_policy", "hardlabelling")
        
        if args.random_state == 42:
            args.random_state = config.get("random_state", 42)
        
        if args.val_frac == 0.2:
            args.val_frac = float(config.get("val_frac", 0.2))
        
        # Optional dataset path overrides
        if args.path_dataset is None:
            args.path_dataset = config.get("dataset", {}).get("path_dataset") if isinstance(config.get("dataset"), dict) else None
        if args.path_gt is None:
            args.path_gt = config.get("dataset", {}).get("path_gt") if isinstance(config.get("dataset"), dict) else None
        if args.path_gt_img is None:
            args.path_gt_img = config.get("dataset", {}).get("path_gt_img") if isinstance(config.get("dataset"), dict) else None
    
    # Validate mode
    if args.mode is None:
        raise ValueError("Mode must be specified (train, test, or labelingSyn)")

    if args.mode == "train":
        dataset = get_dataset(args.dataset, "train", path_dataset=args.path_dataset, path_gt=args.path_gt, path_gt_img=args.path_gt_img)
        cmd_train(args)
        
    elif args.mode == "test":
        dataset = get_dataset(args.dataset, "test", path_dataset=args.path_dataset, path_gt=args.path_gt, path_gt_img=args.path_gt_img)
        cmd_test(args)
        
    else:
        cmd_labeling_syn(args)


if __name__ == "__main__":
    main()

# Backward-compatible API: function used by older callers and package __init__
def run_stage_c(
    config_path: str | None = None,
    mode: str = "train",
    dataset: str = "RAPzs",
    prompting: str = "fixed-rule",
    score_name: str = "blip",
    strategy: str = "identity",
    threshold: float = 0.5,
    classifier: str = "threshold",
    bayes_mode: str = "gauss",
    lora_dir: str | None = None,
):
    # Try to parse YAML config if provided and PyYAML is available
    if config_path and os.path.isfile(config_path):
        try:
            import yaml  # type: ignore

            with open(config_path, "r") as f:
                cfg = yaml.safe_load(f) or {}
            mode = cfg.get("mode", mode)
            dataset = cfg.get("dataset", dataset)
            prompting = cfg.get("prompting", prompting)
            score_name = cfg.get("score_name", score_name)
            strategy = cfg.get("strategy", strategy)
            threshold = float(cfg.get("threshold", threshold))
            classifier = cfg.get("classifier", classifier)
            bayes_mode = cfg.get("bayes_mode", bayes_mode)
            lora_dir = cfg.get("lora_dir", lora_dir)
        except Exception:
            # Best-effort: ignore config parsing issues and proceed with defaults/args
            
            pass

    class _Args:
        def __init__(self):
            self.mode = mode
            self.dataset = dataset
            self.prompting = prompting
            self.score_name = score_name
            self.strategy = strategy
            self.threshold = threshold
            self.classifier = classifier
            self.bayes_mode = bayes_mode
            self.lora_dir = lora_dir

    _args = _Args()
    print("hola")
    if _args.mode == "train":
        
        stats = cmd_train(_args)



    elif _args.mode == "test":
        cmd_test(_args)
    elif _args.mode == "labelingSyn":
        cmd_labeling_syn(_args)
    else:
        raise ValueError(f"Unknown mode: {_args.mode}")

    print("jaja")

