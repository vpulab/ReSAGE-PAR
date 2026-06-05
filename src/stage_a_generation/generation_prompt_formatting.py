"""Prompt formatting utilities with dataset mapping.

Provides `PromptGenerator` which produces text prompts from attribute vectors
delegating to the corresponding dataset implementation (RAPzs, RAPv1, RAPv2,
PETA, PETAzs, PA100k). Uses explicit module/class mappings to resolve datasets.
"""

from __future__ import annotations

from typing import Iterable, Optional
import numpy as np
import importlib
from pathlib import Path
import importlib.util
DATASET_NAME_TO_MODULE_CLASS = {
    # Prefer the 'datasetXxxAll' modules in lora_training/customDatasets
    "RAPzs": ("lora_training.customDatasets.datasetRAPzsAll", "RAPzsDatasetAll"),
    "RAPv1": ("lora_training.customDatasets.datasetRAPv1All", "RAPv1DatasetAll"),
    "RAPv2": ("lora_training.customDatasets.datasetRAPv2All", "RAPv2DatasetAll"),
    "PETA": ("lora_training.customDatasets.datasetPETAAll", "PETADatasetAll"),
    "PETAzs": ("lora_training.customDatasets.datasetPETAzsAll", "PETAzsDatasetAll"),
    "PA100k": ("lora_training.customDatasets.datasetPA100kAll", "PA100kDatasetAll"),
}


def _import_module_fallback(module_path: str):
    """Try several import paths, then fall back to direct file import.

    - Attempts typical package paths (with and without leading 'src.').
    - If those fail, tries to load the module file from
      src/lora_training/customDatasets/<basename>.py next to this repo.
    """
    candidates = [
        module_path,
        module_path.replace("lora_training.", "src.lora_training."),
        module_path.replace("lora_training.", ""),
        module_path.split("lora_training.")[-1],
    ]
    tried = []
    for cand in candidates:
        if not cand or cand in tried:
            continue
        tried.append(cand)
        try:
            return importlib.import_module(cand)
        except Exception:
            pass

    # Filesystem fallback
    try:
        base_name = module_path.split(".")[-1]
        src_dir = Path(__file__).resolve().parents[1]  # points to src/
        module_file = src_dir / "lora_training" / "customDatasets" / f"{base_name}.py"
        if module_file.exists():
            spec = importlib.util.spec_from_file_location(base_name, str(module_file))
            mod = importlib.util.module_from_spec(spec)
            assert spec and spec.loader
            spec.loader.exec_module(mod)
            return mod
    except Exception:
        pass

    raise ImportError(f"Could not import module via candidates: {candidates}")


class PromptGenerator:
    """Generate a textual prompt from an attribute vector using dataset logic.

    Args:
        type: prompt generation type, only 'fixed-rule' supported.
        dataset: one of the keys in DATASET_NAME_TO_MODULE_CLASS (default RAPzs).
        dataset_class/pathDataset/num_images/seed are accepted for API
                 compatibility but not used directly here.
    """

    def __init__(self, type: str = "fixed-rule", dataset: Optional[str] = "RAPzs", dataset_class: Optional[str] = None, pathDataset: Optional[str] = None, num_images: Optional[int] = None, seed: Optional[int] = None):
        self.type = type
        self.dataset_name = dataset or "RAPzs"
        self.dataset = None
        self._load_dataset(self.dataset_name)

    def generatePrompt(self, vector: Iterable, filename: Optional[str] = None) -> str:
        """Return a prompt string for the given attribute vector.

        For `fixed-rule` the method will:
          - if a dataset with `generatePrompt` is available, call it with the
            provided label vector and return the result;
          - otherwise fall back to a simple rule that joins attribute names
            for indices where vector is truthy.

        Other `type` values are placeholders and raise NotImplementedError.
        """
        arr = np.asarray(vector)
        if arr.ndim != 1:
            raise ValueError("vector must be 1D")

        if self.type == "fixed-rule":
            if self.dataset is not None and hasattr(self.dataset, "generatePrompt"):
                res = self.dataset.generatePrompt(list(arr))
                # dataset methods may return (prompt, vector) or just prompt
                if isinstance(res, tuple):
                    prompt = res[0]
                else:
                    prompt = res
                return prompt if isinstance(prompt, str) else ""
            return ""

        elif self.type == "template":
            # placeholder for template-based composition
            raise NotImplementedError("template prompt type not implemented yet")

        else:
            raise ValueError(f"Unknown prompt generator type: {self.type}")

    def _load_dataset(self, dataset_key: str) -> None:
        if dataset_key not in DATASET_NAME_TO_MODULE_CLASS:
            raise ValueError(f"Unsupported dataset '{dataset_key}'. Expected one of: {list(DATASET_NAME_TO_MODULE_CLASS.keys())}")
        module_path, class_name = DATASET_NAME_TO_MODULE_CLASS[dataset_key]
        module = _import_module_fallback(module_path)
        if not hasattr(module, class_name):
            raise ImportError(f"Module '{module.__name__}' has no class '{class_name}'")
        DatasetCls = getattr(module, class_name)
        # instantiate with common default
        try:
            self.dataset = DatasetCls(split="train")
        except Exception:
            self.dataset = DatasetCls()

        


__all__ = ["PromptGenerator"]
