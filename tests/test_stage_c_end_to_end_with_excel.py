import sys
import types

import numpy as np
import pandas as pd
import pytest


def _stub_text_image_similarity_module():
    """Stub out heavy Stage B deps so we can import save_scores_xlsx."""
    name = 'src.stage_b_scoring.text_image_representational_similarity'
    if name in sys.modules:
        return
    mod = types.ModuleType(name)

    class Dummy:
        pass

    mod.TextImageRepresentationalSimilarity = Dummy
    sys.modules[name] = mod


def test_parse_vector_cell_handles_multiple_formats():
    from src.stage_c_pseudolabeling.run_stage_c import _parse_vector_cell

    assert _parse_vector_cell([1, 2]) == [1.0, 2.0]
    assert _parse_vector_cell(np.array([3, 4])) == [3.0, 4.0]
    assert _parse_vector_cell('[0.1, 0.2]') == [0.1, 0.2]
    assert _parse_vector_cell('0.7') == [0.7]
    assert _parse_vector_cell(None) == []


def _write_scores_xlsx(tmp_path, folder_name: str, attr_names: list[str]):
    """Create scores_train.xlsx, scores_test.xlsx and scores_syn.xlsx in a Stage C compatible folder."""
    _stub_text_image_similarity_module()
    from src.stage_b_scoring.run_stage_b import save_scores_xlsx

    class DummyDataset:
        listAllAttrib = list(attr_names)

    out_dir = tmp_path / folder_name
    # Train
    save_scores_xlsx(
        dataset=DummyDataset(),
        pos=[[0.95], [0.90], [0.85]],
        neg=[[0.05], [0.10], [0.15]],
        img_paths=['/abs/img0.png', '/abs/img1.png', '/abs/img2.png'],
        out_dir=str(out_dir),
        train=True,
        is_syn=False,
        prompt_pos=[['pos [strategy=identity 1/1]']] * 3,
        prompt_neg=[['neg [strategy=identity 1/1]']] * 3,
        num_attr_pos=[1, 1, 1],
        num_attr_neg=[0, 0, 0],
        vector_pos=[[1, 0], [1, 0], [1, 0]],
        vector_neg=[[0, 1], [0, 1], [0, 1]],
    )

    # Test
    save_scores_xlsx(
        dataset=DummyDataset(),
        pos=[[0.70], [0.30]],
        neg=[[0.20], [0.60]],
        img_paths=['/abs/t0.png', '/abs/t1.png'],
        out_dir=str(out_dir),
        train=False,
        is_syn=False,
        prompt_pos=[['tpos [strategy=identity 1/1]']] * 2,
        prompt_neg=[['tneg [strategy=identity 1/1]']] * 2,
        num_attr_pos=[1, 1],
        num_attr_neg=[0, 0],
        vector_pos=[[1, 0], [0, 1]],
        vector_neg=[[0, 1], [1, 0]],
    )

    # Synthetic
    save_scores_xlsx(
        dataset=DummyDataset(),
        pos=[[0.80], [0.10]],
        neg=[[0.20], [0.90]],
        img_paths=['syn0', 'syn1'],
        out_dir=str(out_dir),
        train=True,
        is_syn=True,
        prompt_pos=[['syn [strategy=identity 1/1]']] * 2,
        prompt_neg=[['synneg [strategy=identity 1/1]']] * 2,
        num_attr_pos=[1, 1],
        num_attr_neg=[0, 0],
        vector_pos=[[1, 0], [0, 1]],
        vector_neg=[[0, 1], [1, 0]],
    )

    return out_dir


def test_stage_c_train_test_labelingSyn_end_to_end(tmp_path, monkeypatch):
    # Stage C expects specific folder naming based on args
    dataset = 'DUMMY'
    prompting = 'fixed-rule'
    score_name = 'clip'
    strategy = 'identity'
    exp_folder = f'{dataset}_{prompting}_{score_name}_{strategy}_scores'

    _write_scores_xlsx(tmp_path, exp_folder, ['hat', 'bag'])
    monkeypatch.chdir(tmp_path)

    from src.stage_c_pseudolabeling.run_stage_c import cmd_train, cmd_test, cmd_labeling_syn

    args = types.SimpleNamespace(
        dataset=dataset,
        prompting=prompting,
        score_name=score_name,
        strategy=strategy,
        threshold=0.5,
        classifier='bayes',
        bayes_mode='gauss',
    )

    cmd_train(args)

    # Artifacts path should now exist
    artifacts = f'{dataset}_{prompting}_{score_name}_{strategy}_bayes-gauss_si/artifacts'
    clf_path = tmp_path / artifacts / 'classifier.pkl'
    assert clf_path.exists()
    assert (tmp_path / artifacts / 'train_predictions.csv').exists()

    cmd_test(args)
    assert (tmp_path / artifacts / 'test_predictions.csv').exists()

    cmd_labeling_syn(args)
    assert (tmp_path / artifacts / 'pseudolabels_syn.csv').exists()

    # Quick sanity: pseudolabels should have columns imgpath + 2 attrs
    df = pd.read_csv(tmp_path / artifacts / 'pseudolabels_syn.csv')
    assert list(df.columns) == ['imgpath', 'hat', 'bag']


@pytest.mark.xfail(raises=TypeError, reason='ThresholdClassifier.predict does not accept thresh=... in executeTrainIdentityStrategy')
def test_stage_c_threshold_classifier_signature_bug(tmp_path, monkeypatch):
    dataset = 'DUMMY'
    prompting = 'fixed-rule'
    score_name = 'clip'
    strategy = 'identity'
    exp_folder = f'{dataset}_{prompting}_{score_name}_{strategy}_scores'
    _write_scores_xlsx(tmp_path, exp_folder, ['hat', 'bag'])
    monkeypatch.chdir(tmp_path)

    from src.stage_c_pseudolabeling.run_stage_c import cmd_train

    args = types.SimpleNamespace(
        dataset=dataset,
        prompting=prompting,
        score_name=score_name,
        strategy=strategy,
        threshold=0.5,
        classifier='threshold',
        bayes_mode='gauss',
    )
    cmd_train(args)
