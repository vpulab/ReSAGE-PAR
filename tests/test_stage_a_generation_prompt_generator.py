import numpy as np

from src.stage_a_generation.generation_prompt_formatting import PromptGenerator


def _install_dummy_dataset_module(install_module, module_path: str, class_name: str, behavior: str = "str"):
    """Install a dummy dataset module under module_path with class_name.

    behavior:
      - 'str' -> generatePrompt returns a string
      - 'tuple' -> returns (string, vector)
    """
    mod = install_module(module_path)

    class DummyDataset:
        def __init__(self, *args, **kwargs):
            pass

        def generatePrompt(self, vec):
            if behavior == "tuple":
                return (f"prompt_{list(vec)}", vec)
            return f"prompt_{list(vec)}"

    setattr(mod, class_name, DummyDataset)


def test_prompt_generator_returns_string_prompt(tmp_path, install_module, monkeypatch):
    # PromptGenerator expects to import this module for RAPzs
    _install_dummy_dataset_module(
        install_module,
        "lora_training.customDatasets.datasetRAPzsAll",
        "RAPzsDatasetAll",
        behavior="str",
    )

    pg = PromptGenerator(type="fixed-rule", dataset="RAPzs")
    out = pg.generatePrompt([1, 0, 1])
    assert out == "prompt_[np.int64(1), np.int64(0), np.int64(1)]" or out == "prompt_[1, 0, 1]"


def test_prompt_generator_handles_tuple_return(install_module):
    _install_dummy_dataset_module(
        install_module,
        "lora_training.customDatasets.datasetRAPzsAll",
        "RAPzsDatasetAll",
        behavior="tuple",
    )
    pg = PromptGenerator(type="fixed-rule", dataset="RAPzs")
    out = pg.generatePrompt(np.array([0, 1]))
    # Should take the first element of tuple
    assert out.startswith("prompt_")
    assert "0" in out and "1" in out


def test_prompt_generator_validates_vector_1d(install_module):
    _install_dummy_dataset_module(
        install_module,
        "lora_training.customDatasets.datasetRAPzsAll",
        "RAPzsDatasetAll",
        behavior="str",
    )
    pg = PromptGenerator(type="fixed-rule", dataset="RAPzs")
    try:
        pg.generatePrompt(np.zeros((2, 2)))
    except ValueError as e:
        assert "1D" in str(e)
    else:
        raise AssertionError("Expected ValueError")


def test_prompt_generator_rejects_unknown_dataset():
    try:
        PromptGenerator(type="fixed-rule", dataset="NOT_A_DATASET")
    except ValueError as e:
        assert "Unsupported dataset" in str(e)
    else:
        raise AssertionError("Expected ValueError")
