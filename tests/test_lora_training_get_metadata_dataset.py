import json
from pathlib import Path

import numpy as np

from src.lora_training.getMetadataDataset import generate_metadata


def _read_jsonl(path: Path):
    rows = []
    for line in path.read_text().splitlines():
        rows.append(json.loads(line))
    return rows


def test_generate_metadata_uses_get_metadata_by_idx(tmp_path, install_module):
    # Create a fake module + dataset class
    mod = install_module("customDatasets.fake_meta")

    class FakeDataset:
        def __init__(self, **kwargs):
            self.pathToImages = str(tmp_path)
            self.all_images = ["a.png", "b.png"]

        def get_metadata_by_idx(self, idx):
            return {"prompt": f"prompt_{idx}", "name": "person"}

        def __len__(self):
            return len(self.all_images)

    mod.FakeDataset = FakeDataset

    out = Path(tmp_path / "metadata.jsonl")
    out_path = generate_metadata(
        module_name="customDatasets.fake_meta",
        class_name="FakeDataset",
        output=str(out),
        save_vectors=False,
    )

    assert Path(out_path).exists()
    rows = _read_jsonl(out)
    assert len(rows) == 2
    assert rows[0]["file_name"] == "a.png"
    assert rows[0]["text"] == "prompt_0"


def test_generate_metadata_getPrompt_tuple_saves_vector(tmp_path, install_module):
    mod = install_module("customDatasets.fake_prompt")

    class FakeDataset:
        def __init__(self, **kwargs):
            self.pathToImages = str(tmp_path)
            self.all_images = ["x.png"]

        def getPrompt(self, idx):
            return ("hello world", np.array([1, 0, 1]))

        def __len__(self):
            return 1

    mod.FakeDataset = FakeDataset

    out = Path(tmp_path / "metadata.jsonl")
    generate_metadata(
        module_name="customDatasets.fake_prompt",
        class_name="FakeDataset",
        output=str(out),
        save_vectors=True,
    )

    rows = _read_jsonl(out)
    assert rows == [
        {"file_name": "x.png", "text": "hello world", "vector": [1, 0, 1]}
    ]


def test_generate_metadata_fallback_labelsGT_and_generatePrompt(tmp_path, install_module):
    mod = install_module("customDatasets.fake_fallback")

    class FakeDataset:
        def __init__(self, **kwargs):
            self.pathToImages = str(tmp_path)
            self.all_images = ["img0.png", "img1.png"]
            self.filenamesPkl = ["img0.png", "img1.png"]
            self.labelsGT = [np.array([0, 1]), np.array([1, 1])]
            self.class_names = ["person"]

        def generatePrompt(self, label_vec):
            # label_vec can be numpy
            v = np.asarray(label_vec).astype(int).tolist()
            return f"prompt_{v}"

        def __len__(self):
            return len(self.all_images)

    mod.FakeDataset = FakeDataset

    out = Path(tmp_path / "metadata.jsonl")
    generate_metadata(
        module_name="customDatasets.fake_fallback",
        class_name="FakeDataset",
        output=str(out),
        save_vectors=True,
    )

    rows = _read_jsonl(out)
    assert len(rows) == 2
    assert rows[0]["text"] == "prompt_[0, 1]"
    assert rows[0]["vector"] == [0, 1]


def test_generate_metadata_raises_if_class_missing(tmp_path, install_module):
    install_module("customDatasets.fake_missing")

    try:
        generate_metadata(
            module_name="customDatasets.fake_missing",
            class_name="Nope",
            output=str(tmp_path / "m.jsonl"),
        )
    except AttributeError as e:
        assert "has no attribute" in str(e)
    else:
        raise AssertionError("Expected AttributeError")
