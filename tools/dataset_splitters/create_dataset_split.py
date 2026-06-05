#!/usr/bin/env python3

import argparse
import os
import sys
import pickle
from typing import List, Dict, Any
import shutil


SUPPORTED_DATASETS = ["PA100k", "PETA", "PETAzs", "RAPv1", "RAPv2", "RAPzs"]


def _is_int_list(x):
    return isinstance(x, (list, tuple)) and all(isinstance(i, (int,)) for i in x)


def _is_str_list(x):
    return isinstance(x, (list, tuple)) and all(isinstance(i, (str,)) for i in x)


def _idx_to_names(idxs, names: List[str]) -> List[str]:
    out = []
    for i in idxs:
        try:
            out.append(str(names[int(i)]))
        except Exception:
            continue
    return out


def _resolve_split_map(pkl: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    Return a map {split_name: [filenames]} from a dataset_all.pkl-like dict.
    Recognizes keys such as partition/split dicts, *_index, and direct name lists.
    Possible split_name keys: 'train', 'trainval', 'val', 'test'.
    """
    names = pkl.get("image_name") or pkl.get("images")
    if not names:
        raise KeyError("'image_name' not found in pickle; cannot resolve filenames")
    if not isinstance(names, (list, tuple)):
        raise TypeError("'image_name' must be a list")

    split_map: Dict[str, List[str]] = {}

    # 1) partition-like dict
    for key in ("partition", "split", "splits"):
        part = pkl.get(key)
        if isinstance(part, dict):
            for s in ("train", "trainval", "val", "test"):
                v = part.get(s) or (part.get("train_val") if s == "trainval" else None)
                if v is None:
                    continue
                if _is_int_list(v):
                    split_map[s] = _idx_to_names(v, names)
                elif _is_str_list(v):
                    split_map[s] = [str(x) for x in v]

    # 2) explicit index lists
    idx_keys = {
        "train": ["train_index", "train_indices"],
        "trainval": ["trainval_index"],
        "val": ["val_index", "valid_index", "validation_index"],
        "test": ["test_index", "test_indices"],
    }
    for s, keys in idx_keys.items():
        for k in keys:
            v = pkl.get(k)
            if _is_int_list(v):
                split_map[s] = _idx_to_names(v, names)
                break

    # 3) explicit name lists
    name_keys = {
        "train": ["train", "train_list"],
        "trainval": ["trainval", "train_val"],
        "val": ["val", "valid", "validation"],
        "test": ["test", "test_list"],
    }
    for s, keys in name_keys.items():
        for k in keys:
            v = pkl.get(k)
            if _is_str_list(v):
                split_map[s] = [str(x) for x in v]
                break

    return split_map


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _link_or_copy(src: str, dst: str, mode: str):
    if mode == "symlink":
        if os.path.lexists(dst):
            os.remove(dst)
        os.symlink(src, dst)
    elif mode == "hardlink":
        if os.path.exists(dst):
            os.remove(dst)
        os.link(src, dst)
    elif mode == "copy":
        shutil.copy2(src, dst)
    else:
        raise ValueError("mode must be one of: symlink, hardlink, copy")


def main():
    ap = argparse.ArgumentParser(description="Create dataset split folders (train/trainval/val/test) from dataset_all.pkl")
    ap.add_argument("--dataset", required=True, choices=SUPPORTED_DATASETS, help="Dataset name (e.g., PA100k, PETA, RAPv1, ...)")
    ap.add_argument("--pkl", required=True, help="Path to dataset_all.pkl")
    ap.add_argument("--images_root", required=True, help="Path to GT images folder (e.g., .../Rethinking_of_PAR/data/<DATASET>/data)")
    ap.add_argument("--out_root", required=True, help="Output root to create <dataset>/{train|trainval, val, test}")
    ap.add_argument("--mode", choices=["symlink", "hardlink", "copy"], default="symlink", help="Creation mode for files")
    ap.add_argument("--dry_run", action="store_true", help="Print actions without creating files")
    args = ap.parse_args()

    # Load pickle
    with open(args.pkl, "rb") as f:
        pkl = pickle.load(f)
        if not isinstance(pkl, dict):
            print("[WARN] Pickle content is not a dict; proceeding best-effort")

    split_map = _resolve_split_map(pkl)
    if not split_map.get("test") or (not split_map.get("train") and not split_map.get("trainval")):
        print("[ERROR] Could not find required splits in pickle (need test and train or trainval). Aborting.")
        print("       Available splits:", list(split_map.keys()))
        sys.exit(2)

    # Prepare output dirs
    out_dataset = os.path.join(os.path.abspath(args.out_root), args.dataset)
    print(f"[INFO] Output dataset dir: {out_dataset}")

    # Decide which primary training split to use
    if args.dataset == "RAPv1":
        # RAPv1 uses 'trainval' convention
        primary_train_name = "trainval"
        primary_list = split_map.get("trainval") or split_map.get("train") or []
    else:
        primary_train_name = "train"
        primary_list = split_map.get("train") or split_map.get("trainval") or []

    test_list = split_map.get("test", [])
    val_list = split_map.get("val", [])

    print(f"[INFO] {primary_train_name} images: {len(primary_list)} | test images: {len(test_list)} | val images: {len(val_list)}")

    out_train = os.path.join(out_dataset, primary_train_name)
    out_test  = os.path.join(out_dataset, "test")
    out_val   = os.path.join(out_dataset, "val") if val_list else None

    if not args.dry_run:
        _ensure_dir(out_train)
        _ensure_dir(out_test)
        if out_val:
            _ensure_dir(out_val)

    # Create entries
    missing = 0

    def process(names: List[str], out_dir: str) -> int:
        nonlocal missing
        created = 0
        for nm in names:
            src = os.path.join(args.images_root, nm)
            if not os.path.isfile(src):
                missing += 1
                continue
            dst = os.path.join(out_dir, nm)
            if args.dry_run:
                print(f"{args.mode}: {src} -> {dst}")
            else:
                _ensure_dir(os.path.dirname(dst))
                _link_or_copy(src, dst, args.mode)
                created += 1
        return created

    created_train = process(primary_list, out_train)
    created_test  = process(test_list, out_test)
    created_val   = process(val_list, out_val) if out_val else 0

    print(f"[DONE] {primary_train_name}: {created_train} | test: {created_test} | val: {created_val} | Missing: {missing}")
    if missing:
        print("[WARN] Some files listed in pickle were not found in images_root.")


if __name__ == "__main__":
    main()
