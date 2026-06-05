import numpy as np

from src.stage_a_generation.attribute_editing import AttributeEditing


def test_attribute_editing_gt_is_identity():
    ae = AttributeEditing(policy="gt", dataset=None)
    v = np.array([0, 1, 0])
    out = ae.produceSelectedAttributes(v)
    assert np.array_equal(out, v)


def test_attribute_editing_backpacks_is_placeholder_identity():
    ae = AttributeEditing(policy="backpacks", dataset=None)
    v = np.array([1, 2, 3])
    out = ae.produceSelectedAttributes(v)
    assert np.array_equal(out, v)


def test_attribute_editing_unknown_policy_raises():
    ae = AttributeEditing(policy="random_k:3", dataset=None)
    try:
        ae.produceSelectedAttributes([0, 1, 0])
    except NotImplementedError as e:
        assert "not implemented" in str(e).lower()
    else:
        raise AssertionError("Expected NotImplementedError")


def test_attribute_editing_requires_1d_vector():
    ae = AttributeEditing(policy="gt", dataset=None)
    try:
        ae.produceSelectedAttributes(np.zeros((2, 2)))
    except ValueError as e:
        assert "1D" in str(e)
    else:
        raise AssertionError("Expected ValueError")
