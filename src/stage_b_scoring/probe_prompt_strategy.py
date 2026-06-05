"""
Probe Prompt Strategy

Provides a simple strategy class to generate a list of probe prompts given
- a ground-truth attribute vector (gt_vector)
- a target attribute vector (target_vector)

The class is dataset-agnostic: it delegates prompt formatting to an existing
`PromptGenerator` (if provided). If no PromptGenerator is given, the class
returns textual descriptions as placeholder strings.

The main method is `generate_probes(gt_vector, target_vector, ...)` which enforces
exactly one active strategy. It returns a list of dicts:
{'strategy': str, 'vector': np.ndarray, 'prompt': str}.

Notes:
- `_make_prompt()` now takes the active strategy into account and appends a
    lightweight placeholder tag (including position/total when relevant).
- Only one strategy can be active at a time. If a list of strategies is
    provided with more than one item, a ValueError is raised.

"""

from typing import List, Optional, Dict, Any
import numpy as np
import importlib
from pathlib import Path
import importlib.util
import json
import os

DATASET_NAME_TO_MODULE_CLASS = {
    "RAPzs": ("lora_training.customDatasets.datasetRAPzsAll", "RAPzsDatasetAll"),
    "RAPv1": ("lora_training.customDatasets.datasetRAPv1All", "RAPv1DatasetAll"),
    "RAPv2": ("lora_training.customDatasets.datasetRAPv2All", "RAPv2DatasetAll"),
    "PETA": ("lora_training.customDatasets.datasetPETAAll", "PETADatasetAll"),
    "PETAzs": ("lora_training.customDatasets.datasetPETAzsAll", "PETAzsDatasetAll"),
    "PA100k": ("lora_training.customDatasets.datasetPA100kAll", "PA100kDatasetAll"),
}


def _import_module_fallback(module_path: str):
    """Try several import paths, then fall back to direct file import."""
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


class ProbePromptStrategy:
    """Generate probe prompts from attribute vectors.

    Args:
        prompt_generator: optional instance with `generatePrompt(vector)` method.
            If provided, it's used to convert an attribute vector into a text prompt.
        float_threshold: threshold to binarize float-valued attribute vectors when
            strategies require binary decisions (default 0.5).
    """

    def __init__(self, prompt_generator: Optional[Any] = None, float_threshold: float = 0.5, strategy: str = "identity", dataset_name: Optional[str] = None):
        self.prompt_generator = prompt_generator
        self.float_threshold = float_threshold
        self.active_strategy = strategy
        self.dataset = None
        self.dataset_name = dataset_name
        # Load dataset if dataset_name is provided
        if dataset_name:
            self._load_dataset(dataset_name)

        if self.active_strategy == "gemini":
            CARPETA_JSONS = "./src/stage_b_scoring/prompting/" # Cambia esto a la carpeta donde los guardes
            positives_json, negatives_json = self.load_probe_prompts(self.dataset_name, prompts_dir=CARPETA_JSONS)
            
            self.positive_prompts = []
            for key in positives_json.keys():
                self.positive_prompts.append(positives_json[key])
            
            self.negative_prompts = []
            for key in negatives_json.keys():
                self.negative_prompts.append(negatives_json[key])

            print(self.positive_prompts)
            print(self.negative_prompts)

    def _load_dataset(self, dataset_key: str) -> None:
        """Load dataset class similar to PromptGenerator."""
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

    def _to_numpy(self, vec) -> np.ndarray:
        if isinstance(vec, np.ndarray):
            return vec.astype(float)
        try:
            return np.array(list(vec), dtype=float)
        except Exception:
            raise ValueError("Unsupported vector type; expected array-like of numbers")

    def _binarize(self, vec: np.ndarray) -> np.ndarray:
        return (vec >= self.float_threshold).astype(int)

    def _make_prompt(
        self,
        vector: np.ndarray,
        strategy: str,
        position: Optional[int] = None,
        total: Optional[int] = None,
        fallback_prefix: str = "Attributes:",
    ) -> str:
        base: str | None = None
        if self.prompt_generator is not None:
            try:
                base = self.prompt_generator.generatePrompt(vector)
            except Exception:
                base = None

        # Build a small placeholder suffix that encodes strategy and ordering
        suffix_parts = [f"strategy={strategy}"]
        if position is not None and total is not None:
            suffix_parts.append(f"{position}/{total}")
        suffix = " [" + " ".join(suffix_parts) + "]"

        if base:
            return base + suffix

        # Fallback textual description with active indices
        vec_bin = self._binarize(vector)
        active = [str(i) for i, v in enumerate(vec_bin) if v]
        return f"{fallback_prefix} active_indices={','.join(active)}" + suffix



    def load_probe_prompts(self, dataset_name, prompts_dir="."):
        """
        Carga los diccionarios JSON de prompts positivos y negativos para un dataset.
        
        Args:
            dataset_name (str): Nombre del dataset (ej: 'PETAzs', 'RAPv2', 'PA100k').
            prompts_dir (str): Carpeta donde están guardados los .json.
            
        Returns:
            tuple: (probes, probes_neg) diccionarios con los prompts.
        """
        # 1. Normalizar el nombre a minúsculas (ej: "PETAzs" -> "petazs")
        dataset_clean = dataset_name.lower()
        
        # 2. Construir las rutas de los archivos
        pos_file_path = os.path.join(prompts_dir, f"{dataset_clean}_gemini.json")
        neg_file_path = os.path.join(prompts_dir, f"{dataset_clean}_gemini_negative.json")
        
        probes = {}
        probes_neg = {}
        
        # 3. Cargar el JSON de prompts Positivos
        if os.path.exists(pos_file_path):
            with open(pos_file_path, 'r', encoding='utf-8') as f:
                probes = json.load(f)
            print(f"✅ Cargados {len(probes)} prompts POSITIVOS desde: {pos_file_path}")
        else:
            print(f"⚠️ Atención: No se encontró el archivo {pos_file_path}")
            
        # 4. Cargar el JSON de prompts Negativos
        if os.path.exists(neg_file_path):
            with open(neg_file_path, 'r', encoding='utf-8') as f:
                probes_neg = json.load(f)
            print(f"✅ Cargados {len(probes_neg)} prompts NEGATIVOS desde: {neg_file_path}")
        else:
            print(f"⚠️ Atención: No se encontró el archivo {neg_file_path}")
        
        return probes, probes_neg



    def generate_probes(
        self,
        gt_vector,
        target_vector,
        n_probes: Optional[int] = None,
        strategies: Optional[List[str]] = None,
        strategy: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Generate a list of probe prompts.

        Parameters:
        - gt_vector: ground-truth attribute vector (array-like)
        - target_vector: attribute vector representing desired/target attributes (array-like)
        - n_probes: optional cap on how many probes to return (if a strategy
          yields more, results are truncated; if None, all generated are used).
        - strategies: legacy list-form input. If provided, it must contain
          exactly one item; otherwise a ValueError is raised.
        - strategy: preferred single strategy string (recommended). Supported:
            - 'identity': prompt from gt_vector
            - 'target': prompt from target_vector
            - 'diff_target': keep only attributes that differ between gt and target (use target's values)
            - 'swap_diff': flip the attributes that differ between gt and target
            - 'interpolate:k': interpolate k steps between gt and target and return prompts
            - 'topk_target:k': set only top-k values (by magnitude) of target_vector to 1

        Returns:
          List of dicts: {'strategy': str, 'vector': np.ndarray, 'prompt': str}
        """
        gt = self._to_numpy(gt_vector)
        tgt = self._to_numpy(target_vector)
        if gt.shape != tgt.shape:
            raise ValueError("gt_vector and target_vector must have the same shape")

        

        # Build candidate vectors for the active strategy
        vectors: list[np.ndarray] = []
        labels: list[str] = []

        if self.active_strategy == "identity":
            vectors = [gt.copy()]
            labels = ["identity"]
            # Apply n_probes cap if requested
            total = len(vectors)
            use_n = total if n_probes is None else max(0, min(n_probes, total))

            probes: List[Dict[str, Any]] = []
            probes_neg: List[Dict[str, Any]] = []
            for idx in range(use_n):
                vec = vectors[idx]
                lab = labels[idx]
                prompt = self._make_prompt(vec, strategy=self.active_strategy, position=idx + 1, total=use_n, fallback_prefix=lab.capitalize())
                probes.append({"strategy": lab, "vector": vec, "prompt": prompt})
                
                
                vectorsSel = self.dataset.getVectorCompPercentageAttributes([vec], percentageToChange=1.0)
                
                if len(vectorsSel) == 0:
                    return None, None
                else:
                    vectorsSel = vectorsSel[0]
                    vectorsSel = np.array(vectorsSel)
                    
                promptNew = self._make_prompt(vectorsSel, strategy=self.active_strategy, position=idx + 1, total=use_n, fallback_prefix=lab.capitalize())
                probes_neg.append({"strategy": lab, "vector": vectorsSel, "prompt": promptNew})

            return probes, probes_neg
        
        elif self.active_strategy == "gemini":
            
            vectors = [gt.copy()]
 
            
            probes = []
            probes_neg = []

            vectorsSel = self.dataset.getVectorCompPercentageAttributes([vectors[0]], percentageToChange=1.0)

            if len(vectorsSel) == 0:
                    return None, None
            else:
                vectorsSel = vectorsSel[0]
                vectorsSel = np.array(vectorsSel)

            probes = [{"strategy": "gemini", "vector": vectors[0], "prompt": pos} for pos in self.positive_prompts]
            probes_neg = [{"strategy": "gemini", "vector": vectorsSel, "prompt": neg} for neg in self.negative_prompts]
            
            return probes, probes_neg
        else:
            raise ValueError(f"Unknown strategy '{self.active_strategy}'")

        

       

