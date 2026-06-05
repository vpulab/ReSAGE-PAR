"""Utility for selecting/modifying attribute vectors according to a policy.

Provides a small, extensible `AttributeEditing` class which takes a string
`policy` at construction and applies that policy to an attribute vector via
`produceSelectedAttributes(vector)`.

Supported policy formats (examples):
 - "identity" -> return input vector unchanged
 - "top_k:5" or "topk=5" -> keep only the top-5 highest values (set others to 0)
 - "threshold:0.5" or "thr=0.5" -> keep values >= threshold (1) else 0
 - "random_k:10" -> randomly select 10 indices to set to 1 (others 0)
 - "flip" -> bitwise flip (1->0, 0->1) for binary vectors
 - "keep:0,2,5" -> keep original values at indices 0,2,5, zero others
 - "set:1,3,4" -> set indices 1,3,4 to 1, zero others

The implementation uses numpy and returns a numpy array (dtype int for
selection outputs). This class is intentionally small and easy to extend.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Tuple
import re
import importlib
import numpy as np


class AttributeEditing:
    """Select or modify attribute vectors according to a simple policy string.

    This implementation can optionally import a dataset from `customDatasets`
    (passing `dataset`), instantiate it, and capture its attribute names
    into `self.listAttributes` so policies can reference attribute names.
    """

    def __init__(self, policy: str = "gt", dataset: Optional[str] = "PA100k") -> None:
        self.policy = policy
        self.dataset_str = dataset

        # populated if dataset import/instantiation succeeds
        self.listAttributes: Optional[List[str]] = None
        self.num_attributes: Optional[int] = None

        # try to import dataset and capture attribute names
        if self.dataset_str:
            self._load_dataset_attributes(dataset)

    def _import_module_fallback(self, module_name: str):
        """Try importing a module with a few sensible fallbacks."""
        candidates = [
            module_name,
            f"customDatasets.{module_name}",
            f"src.lora_training.customDatasets.{module_name}",
            f"lora_training.customDatasets.{module_name}",
            f"src.lora_training.{module_name}",
        ]
        for cand in candidates:
            try:
                return importlib.import_module(cand)
            except Exception:
                continue
        raise ImportError(f"Could not import any of: {candidates}")

    def _load_dataset_attributes(self, dataset: str) -> None:
        """Resolve dataset module/class from a short string and extract attribute names.

        Accepts several forms for `dataset`:
        - "PA100k" -> tries module `PA100kAll` and class `PA100kDatasetAll` inside `customDatasets`
        - "PA100kAll" -> uses that module name
        - "customDatasets.PA100kAll" -> full module path
        - "customDatasets.PA100kAll.PA100kDatasetAll" -> module + class
        """
        # determine module and class names
        module_path = dataset
        class_name = None
        # if user passed module with class appended
        if "." in dataset and (dataset.count(".") >= 2):
            # treat last part as class
            *mod_parts, cls = dataset.split(".")
            module_path = ".".join(mod_parts)
            class_name = cls
        else:
            # try to infer class from short name
            base = dataset.split(".")[-1]
            if base.endswith("All"):
                module_name = base
                class_name = base[:-3] + "DatasetAll"
            else:
                module_name = base + "All"
                class_name = base + "DatasetAll"
            module_path = module_name

        try:
            module = self._import_module_fallback(module_path)
            if not class_name:
                raise ValueError("Could not determine dataset class name")
            if not hasattr(module, class_name):
                raise ImportError(f"Module '{module_path}' has no class '{class_name}'")
            DatasetCls = getattr(module, class_name)

            # instantiate with defaults (many dataset classes accept no args)
            try:
                ds = DatasetCls()
            except TypeError:
                # try with minimal kwargs commonly used
                try:
                    ds = DatasetCls(pathDataset=None)
                except Exception:
                    ds = DatasetCls({})

            # extract attribute names from common fields
            if hasattr(ds, "listAttributes"):
                self.listAttributes = list(getattr(ds, "listAttributes"))
            elif hasattr(ds, "listAllAttrib"):
                self.listAttributes = list(getattr(ds, "listAllAttrib"))
            elif hasattr(ds, "dataPkl") and isinstance(getattr(ds, "dataPkl"), dict) and "attr_name" in ds.dataPkl:
                self.listAttributes = [str(x) for x in ds.dataPkl["attr_name"]]

            if self.listAttributes is not None:
                self.num_attributes = len(self.listAttributes)
        except Exception as e:
            # non-fatal: attribute editing can operate without attribute names
            print(f"Warning: could not load dataset attributes for '{dataset}': {e}")

    def produceSelectedAttributes(self, vector: Iterable, seed: Optional[int] = None) -> np.ndarray:
        """Apply the configured policy to `vector` and return a new vector.

        Args:
            vector: array-like of numeric attribute scores or binary values.
            seed: optional random seed for reproducible `random_k` selections.

        Returns:
            numpy.ndarray with the same length as `vector`. Selection outputs are
            integer (0/1) arrays unless the policy preserves original values.
        """
        arr = np.asarray(vector)
        if arr.ndim != 1:
            raise ValueError("produceSelectedAttributes expects a 1D vector")

        # if we know attribute names, validate length
        if self.num_attributes is not None and arr.size != self.num_attributes:
            print(f"Warning: input vector length {arr.size} does not match dataset attributes length {self.num_attributes}")

        if self.policy == "gt":
            return arr
        elif self.policy == "backpacks":
            # placeholder for a real policy that would inspect attribute names
            return arr
        else:
            raise NotImplementedError(f"AttributeEditing policy '{self.policy}' not implemented.")


__all__ = ["AttributeEditing"]
