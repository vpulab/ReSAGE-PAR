import numpy as np
from easydict import EasyDict

def get_pedestrian_metrics(
    gt_label,
    preds_probs,
    threshold=0.5,
    index=None,
    cfg=None,
    ignore_value=-1,
):
    """
    - Entries where preds_probs == ignore_value (default -1) are excluded
      from ALL metric computations (both numerator & denominator).
    - count_per_label[j] = #samples used for attribute j (not ignored).
    - count_per_instance[i] = #attributes used for sample i (not ignored).
    """
    gt_label = np.asarray(gt_label)
    preds_probs = np.asarray(preds_probs, dtype=float)

    # Valid where not ignored and finite
    valid = np.isfinite(preds_probs) & (preds_probs != ignore_value)
    pred_bin = (preds_probs > threshold)

    if index is not None:
        gt_label  = gt_label[:, index]
        pred_bin  = pred_bin[:, index]
        valid     = valid[:, index]

    def _safe_div(num, den):
        """num/den with where-mask, returns NaN where den<=0 or non-finite."""
        num = np.asarray(num, dtype=float)
        den = np.asarray(den, dtype=float)
        out = np.full_like(num, np.nan, dtype=float)
        mask = (den > 0) & np.isfinite(den) & np.isfinite(num)
        with np.errstate(invalid='ignore', divide='ignore'):
            out[mask] = num[mask] / den[mask]
        return out

    # ---------- Per-label (column-wise) with masking ----------
    gt_pos = np.sum((gt_label == 1) & valid, axis=0).astype(float)  # TP+FN
    gt_neg = np.sum((gt_label == 0) & valid, axis=0).astype(float)  # TN+FP

    tp = np.sum((gt_label == 1) & (pred_bin == 1) & valid, axis=0).astype(float)
    tn = np.sum((gt_label == 0) & (pred_bin == 0) & valid, axis=0).astype(float)
    fp = np.sum((gt_label == 0) & (pred_bin == 1) & valid, axis=0).astype(float)
    fn = np.sum((gt_label == 1) & (pred_bin == 0) & valid, axis=0).astype(float)

    tpr = _safe_div(tp, gt_pos)   # recall for positives
    tnr = _safe_div(tn, gt_neg)   # recall for negatives

    # Mean of available parts only (avoid "mean of empty slice"):
    have = np.isfinite(tpr).astype(float) + np.isfinite(tnr).astype(float)
    label_ma = np.where(have > 0,
                        np.nan_to_num(tpr) + np.nan_to_num(tnr),
                        np.nan)
    label_ma = label_ma / np.where(have > 0, have, np.nan)

    label_prec = _safe_div(tp, tp + fp)
    label_acc  = _safe_div(tp + tn, tp + tn + fp + fn)
    label_f1   = _safe_div(2 * label_prec * tpr, (label_prec + tpr))

    # Global mean accuracy across labels (ignore NaNs)
    ma = np.nanmean(label_ma) if np.isfinite(label_ma).any() else np.nan

    # ---------- Instance-level (row-wise) with masking ----------
    gt_pos_row   = np.sum((gt_label == 1) & valid, axis=1).astype(float)
    pred_pos_row = np.sum((pred_bin == 1) & valid, axis=1).astype(float)
    intersect    = np.sum((gt_label == 1) & (pred_bin == 1) & valid, axis=1).astype(float)

    # "Union" per your original definition (counts intersection twice)
    #union = ((gt_label == 1).astype(int) + (pred_bin == 1).astype(int)) * valid.astype(int)
    #union = np.sum(union, axis=1).astype(float)

    #inst_acc = _safe_div(intersect, union)
    
    union = np.sum(((gt_label == 1) | (pred_bin == 1)) & valid, axis=1).astype(float)
    inst_acc = _safe_div(intersect, union)
    
    inst_prec = _safe_div(intersect, pred_pos_row)
    inst_recall = _safe_div(intersect, gt_pos_row)
    inst_f1 = _safe_div(2 * inst_prec * inst_recall, (inst_prec + inst_recall))

    result = EasyDict()
    result.label_pos_recall = tpr
    result.label_neg_recall = tnr
    result.label_prec = label_prec
    result.label_acc  = label_acc
    result.label_f1   = label_f1
    result.label_ma   = label_ma
    result.ma         = ma

    result.count_per_label    = np.sum(valid, axis=0).astype(int)
    result.count_per_instance = np.sum(valid, axis=1).astype(int)

    result.instance_acc_label    = inst_acc
    result.instance_prec_label   = inst_prec
    result.instance_recall_label = inst_recall
    result.instance_f1_label     = inst_f1

    # Means across instances (ignore NaNs)
    result.instance_acc    = float(np.nanmean(inst_acc))    if np.isfinite(inst_acc).any()    else np.nan
    result.instance_prec   = float(np.nanmean(inst_prec))   if np.isfinite(inst_prec).any()   else np.nan
    result.instance_recall = float(np.nanmean(inst_recall)) if np.isfinite(inst_recall).any() else np.nan
    result.instance_f1     = _safe_div(2 * result.instance_prec * result.instance_recall,
                                       (result.instance_prec + result.instance_recall))

    # Error counts per label (masked)
    result.error_num = fp + fn
    result.fn_num    = fn
    result.fp_num    = fp

    return result