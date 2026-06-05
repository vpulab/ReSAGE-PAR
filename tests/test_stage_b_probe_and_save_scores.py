import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd


def _stub_text_image_similarity_module():
    """Install a stub for src.stage_b_scoring.text_image_representational_similarity
    so importing run_stage_b doesn't require transformers.
    """
    name = "src.stage_b_scoring.text_image_representational_similarity"
    if name in sys.modules:
        return
    mod = types.ModuleType(name)
    class Dummy:
        pass
    mod.TextImageRepresentationalSimilarity = Dummy
    sys.modules[name] = mod


def test_save_scores_xlsx_creates_sheets_and_interleaved_columns(tmp_path):
    _stub_text_image_similarity_module()
    from src.stage_b_scoring.run_stage_b import save_scores_xlsx

    class DummyDataset:
        listAllAttrib = ["hat", "bag"]

    out_dir = tmp_path / "exp"
    save_scores_xlsx(
        dataset=DummyDataset(),
        pos=[[0.9], [0.8]],
        neg=[[0.1], [0.2]],
        img_paths=["/abs/path/img0.png", "/abs/path/img1.png"],
        out_dir=str(out_dir),
        train=True,
        is_syn=False,
        prompt_pos=[["a person [strategy=identity 1/1]"], ["b person [strategy=identity 1/1]"]],
        prompt_neg=[["neg [strategy=identity 1/1]"], ["neg2 [strategy=identity 1/1]"]],
        num_attr_pos=[1, 1],
        num_attr_neg=[0, 0],
        vector_pos=[[1, 0], [1, 1]],
        vector_neg=[[0, 1], [0, 0]],
    )

    xlsx = out_dir / "scores_train.xlsx"
    assert xlsx.exists()

    df_prompt = pd.read_excel(xlsx, sheet_name="prompting", engine="openpyxl")
    df_sanity = pd.read_excel(xlsx, sheet_name="sanity_check", engine="openpyxl")

    assert set(["imgpath", "pos", "neg", "prompt_pos", "prompt_neg"]).issubset(df_prompt.columns)

    # sanity_check should include attribute columns in interleaved order
    assert "hat_pos" in df_sanity.columns and "hat_neg" in df_sanity.columns
    assert "bag_pos" in df_sanity.columns and "bag_neg" in df_sanity.columns

    # Strategy suffix should be stripped in prompt_pos
    assert df_prompt.loc[0, "prompt_pos"] == "a person"


def test_probe_prompt_strategy_identity_returns_pos_and_neg(install_module):
    # Install a dummy dataset module for PA100k
    mod = install_module("lora_training.customDatasets.datasetPA100kAll")

    class DummyDS:
        def __init__(self, *args, **kwargs):
            pass

        def getVectorCompPercentageAttributes(self, vecs, percentageToChange=1.0):
            # Return complement vector for first one
            v = np.asarray(vecs[0]).astype(int)
            return [1 - v]

    mod.PA100kDatasetAll = DummyDS

    from src.stage_b_scoring.probe_prompt_strategy import ProbePromptStrategy

    class DummyPromptGen:
        def generatePrompt(self, vector):
            return f"P{np.asarray(vector).astype(int).tolist()}"

    p = ProbePromptStrategy(prompt_generator=DummyPromptGen(), strategy="identity", dataset_name="PA100k")
    pos, neg = p.generate_probes(gt_vector=[0, 1], target_vector=[0, 1])
    assert pos is not None and neg is not None
    assert len(pos) == 1 and len(neg) == 1
    assert "[strategy=identity" in pos[0]["prompt"]
    assert "[strategy=identity" in neg[0]["prompt"]


def test_probe_prompt_strategy_returns_none_when_no_complements(install_module):
    mod = install_module("lora_training.customDatasets.datasetPA100kAll")

    class DummyDS:
        def __init__(self, *args, **kwargs):
            pass

        def getVectorCompPercentageAttributes(self, vecs, percentageToChange=1.0):
            return []

    mod.PA100kDatasetAll = DummyDS

    from src.stage_b_scoring.probe_prompt_strategy import ProbePromptStrategy

    p = ProbePromptStrategy(prompt_generator=None, strategy="identity", dataset_name="PA100k")
    pos, neg = p.generate_probes(gt_vector=[0, 1], target_vector=[0, 1])
    assert pos is None and neg is None
