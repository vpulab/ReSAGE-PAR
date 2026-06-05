#!/usr/bin/env python
"""Generic metadata generator that uses dataset classes in `customDatasets`.

Usage examples (command-line arguments):
  python getMetadataDataset.py --module customDatasets.PA100kAll --class PA100kDatasetAll

Usage examples (YAML config file):
  python getMetadataDataset.py --config configs/stage_a.yaml

This will import the dataset class, instantiate it (using defaults), and write
`metadata.jsonl` into the dataset's `pathToImages` folder (or to `--output`).
"""
import argparse
import importlib
import json
import os
import sys
from typing import Any

try:
    import yaml
except ImportError:
    yaml = None


def import_module_fallback(module_name: str):
    """Try importing module with several common prefixes to be robust."""
    try:
        return importlib.import_module(module_name)
    except Exception:
        candidates = [
            f"src.lora_training.{module_name}",
            f"lora_training.{module_name}",
        ]
        for cand in candidates:
            try:
                return importlib.import_module(cand)
            except Exception:
                continue
    raise ImportError(f"Could not import module '{module_name}' (tried fallbacks).")


def generate_metadata(module_name: str, class_name: str, output: str = None, num_images: int = None, pathDataset: str = None, path_dataset: str = None, path_gt: str = None, path_gt_img: str = None, seed: int = None, save_vectors: bool = False) -> str:
    module = import_module_fallback(module_name)
    if not hasattr(module, class_name):
        raise AttributeError(f"Module '{module_name}' has no attribute '{class_name}'")

    cls = getattr(module, class_name)

    print(num_images)

    # Support both pathDataset (legacy) and path_dataset (new)
    if path_dataset is not None:
        pathDataset = path_dataset

    # Instantiate dataset. Pass provided kwargs if given.
    init_kwargs: dict[str, Any] = {}
    if num_images is not None:
        init_kwargs["numImages"] = num_images
    if pathDataset is not None:
        init_kwargs["pathDataset"] = pathDataset
    if path_dataset is not None:
        init_kwargs["path_dataset"] = path_dataset
    if path_gt is not None:
        init_kwargs["path_gt"] = path_gt
    if path_gt_img is not None:
        init_kwargs["path_gt_img"] = path_gt_img
    if seed is not None:
        init_kwargs["seed"] = seed

    dataset = cls(**init_kwargs)

    # Determine output path
    namemeta = "metadata.jsonl"
    if output is None:
        if hasattr(dataset, "pathToImages"):
            out_dir = dataset.pathToImages
        elif hasattr(dataset, "pathDataset"):
            out_dir = os.path.join(dataset.pathDataset, getattr(dataset, "TRAIN_FOLDER", ""))
        else:
            out_dir = os.getcwd()
        os.makedirs(out_dir, exist_ok=True)
        output = os.path.join(out_dir, namemeta)

    entries = []
    # Prefer dataset-provided list of images or fallback to len(dataset)
    if hasattr(dataset, "all_images") and dataset.all_images:
        total = len(dataset.all_images)
    else:
        try:
            total = len(dataset)
        except Exception:
            total = 0

    for idx in range(total):
        # get filename
        filename = None
        if hasattr(dataset, "all_images") and idx < len(dataset.all_images):
            filename = dataset.all_images[idx]
        elif hasattr(dataset, "get_image_by_idx"):
            # no filename available; generate placeholder
            filename = f"image_{idx}.png"

        # try to get metadata using dataset.get_metadata_by_idx or getPrompt
        meta = None
        vector = None
        if hasattr(dataset, "get_metadata_by_idx"):
            try:
                meta = dataset.get_metadata_by_idx(idx)
            except Exception:
                meta = None

        if meta is None and hasattr(dataset, "getPrompt"):
            try:
                result = dataset.getPrompt(idx)
                # check if getPrompt returns tuple (prompt, vector) or just prompt
                if isinstance(result, tuple) and len(result) == 2:
                    prompt, vector = result
                else:
                    prompt = result
                meta = {"prompt": prompt, "name": getattr(dataset, "class_names", ["person"])[0]}
            except Exception:
                meta = None

        if meta is None and hasattr(dataset, "generatePrompt") and hasattr(dataset, "labelsGT") and hasattr(dataset, "filenamesPkl"):
            try:
                # try to recover label index
                filename_image = dataset.all_images[idx]
                indexToLabel = dataset.filenamesPkl.index(filename_image)
                labelGT = dataset.labelsGT[indexToLabel]
                prompt = dataset.generatePrompt(labelGT)
                meta = {"prompt": prompt, "name": getattr(dataset, "class_names", ["person"])[0]}
                vector = labelGT  # save the label vector
            except Exception:
                meta = None

        if meta is None:
            # fallback to dataset[idx] if it returns a dict with caption
            try:
                item = dataset[idx]
                if isinstance(item, dict) and ("caption" in item or "text" in item or "prompt" in item):
                    caption = item.get("caption") or item.get("text") or item.get("prompt")
                    meta = {"prompt": caption, "name": item.get("name", getattr(dataset, "class_names", ["person"])[0])}
            except Exception:
                meta = None

        if meta is None:
            # give up for this index
            continue

        # If user requested vectors but we haven't got one yet, try additional fallbacks
        if save_vectors and vector is None:
            # 1) try dataset[idx] if it returns a dict containing a vector/label
            try:
                item = dataset[idx]
                if isinstance(item, dict):
                    if "vector" in item:
                        vector = item.get("vector")
                    elif "label" in item:
                        vector = item.get("label")
                    elif "labels" in item:
                        vector = item.get("labels")
            except Exception:
                # ignore failures from indexing
                pass

            # 2) try to recover from dataset.labelsGT + dataset.filenamesPkl using filename
            if vector is None and hasattr(dataset, "labelsGT") and hasattr(dataset, "filenamesPkl") and filename is not None:
                try:
                    indexToLabel = dataset.filenamesPkl.index(filename)
                    vector = dataset.labelsGT[indexToLabel]
                except Exception:
                    vector = None

        entry = {"file_name": filename, "text": meta.get("prompt")}
        # add vector if available and save_vectors is True
        if save_vectors and vector is not None:
            # Convert common array types to plain lists for JSON
            try:
                if isinstance(vector, list):
                    entry["vector"] = vector
                else:
                    entry["vector"] = vector.tolist()
            except Exception:
                # last-resort: stringify
                entry["vector"] = list(vector) if hasattr(vector, "__iter__") else vector

        entries.append(entry)

    # write metadata
    with open(output, "w") as f:
        for item in entries:
            f.write(json.dumps(item) + "\n")

    return output


def _parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Path to YAML config file (takes precedence over CLI args)")
    parser.add_argument("--module", help="Module path for dataset class (e.g. customDatasets.PA100kAll)")
    parser.add_argument("--class", help="Class name inside module (e.g. PA100kDatasetAll)", dest="class_name")
    parser.add_argument("--output", help="Output metadata file path (optional)")
    parser.add_argument("--num-images", type=int, help="Number of images to use (optional)")
    parser.add_argument("--pathDataset", help="Override dataset path if supported by class constructor")
    parser.add_argument("--path-dataset", help="Custom path to dataset images (new parameter, recommended over pathDataset)")
    parser.add_argument("--path-gt", help="Custom path to ground truth pickle file")
    parser.add_argument("--path-gt-img", help="Custom path to ground truth images folder")
    parser.add_argument("--seed", type=int, help="Random seed to pass to dataset constructor (optional)")
    parser.add_argument("--save-vectors", action="store_true", help="Save attribute label vectors in metadata (optional)")
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    
    # Load config from YAML if provided
    if args.config:
        if yaml is None:
            raise ImportError("PyYAML is required to load config files. Install with: pip install pyyaml")
        
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
        
        # Extract getMetadata or dataset section if it exists, otherwise use root config
        if "getMetadata" in config:
            config = config["getMetadata"]
        elif "dataset" in config:
            config = config["dataset"]
        
        # Use config values, falling back to CLI args if provided
        module = config.get("module", args.module)
        class_name = config.get("class", args.class_name)
        output = config.get("output", args.output)
        num_images = config.get("num_images", args.num_images)
        pathDataset = config.get("pathDataset") or config.get("path_dataset", args.path_dataset or args.pathDataset)
        path_dataset = config.get("path_dataset", args.path_dataset)
        path_gt = config.get("path_gt", args.path_gt)
        path_gt_img = config.get("path_gt_img", args.path_gt_img)
        seed = config.get("seed", args.seed)
        save_vectors = config.get("save_vectors", args.save_vectors)
    else:
        # Use CLI args
        module = args.module
        class_name = args.class_name
        output = args.output
        num_images = args.num_images
        pathDataset = args.pathDataset
        path_dataset = args.path_dataset
        path_gt = args.path_gt
        path_gt_img = args.path_gt_img
        seed = args.seed
        save_vectors = args.save_vectors
    
    # Validate required arguments
    if not module or not class_name:
        raise ValueError("Both 'module' and 'class' (or 'class_name') are required. Provide via --config YAML or CLI args.")
    
    out = generate_metadata(module, class_name, output=output, num_images=num_images, pathDataset=pathDataset, 
                           path_dataset=path_dataset, path_gt=path_gt, path_gt_img=path_gt_img, seed=seed, save_vectors=save_vectors)
    print(f"Wrote metadata to: {out}")


if __name__ == "__main__":
    main()
