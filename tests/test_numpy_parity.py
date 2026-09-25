"""NumPy-backend parity tests against the PyTorch/Transformers references.

Mirrors tests/test_model.py, which runs the same comparisons for the MLX
implementation. Skipped when torch/transformers are not installed.
"""

import numpy as np
import pytest

from laya_mlx.config import EncoderConfig, sanitize_weights
from laya_mlx.numpy_model import (
    NumpyDecisionModel,
    NumpyModernBert,
    attention_masks,
)

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")


def torch_config(**kwargs):
    return transformers.ModernBertConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=96,
        num_hidden_layers=3,
        num_attention_heads=1,
        local_attention=128,
        pad_token_id=0,
        bos_token_id=2,
        eos_token_id=3,
        cls_token_id=2,
        sep_token_id=3,
        **kwargs,
    )


@pytest.mark.parametrize("local_theta", [10000.0, 160000.0])
def test_numpy_encoder_matches_transformers_across_window_and_padding(local_theta):
    torch.manual_seed(0)
    cfg = torch_config(
        rope_parameters={
            "full_attention": {"rope_type": "default", "rope_theta": 160000.0},
            "sliding_attention": {"rope_type": "default", "rope_theta": local_theta},
        }
    )
    reference = transformers.ModernBertModel(cfg).eval()
    model = NumpyModernBert(EncoderConfig.from_dict(cfg.to_dict()), prefix="")
    weights = {
        k: v.detach().numpy().astype(np.float32) for k, v in reference.state_dict().items()
    }
    ids = np.random.default_rng(1).integers(1, 128, size=(2, 145)).astype(np.int32)
    mask = np.ones((2, 145), np.int32)
    mask[1, 13:] = 0  # padded queries beyond the local window must remain finite
    with torch.inference_mode():
        expected = reference(
            torch.tensor(ids), attention_mask=torch.tensor(mask)
        ).last_hidden_state.numpy()
    actual = model(ids, mask.astype(bool), weights)
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(
        actual[mask.astype(bool)], expected[mask.astype(bool)], atol=2e-5, rtol=2e-5
    )


def test_numpy_decision_head_matches_torch_transformer_encoder_layer():
    torch.manual_seed(1)
    reference = torch.nn.TransformerEncoderLayer(
        64, 1, 256, batch_first=True, norm_first=True
    ).eval()
    model = NumpyDecisionModel(EncoderConfig.from_dict(torch_config().to_dict()), {})
    weights = sanitize_weights(
        {k: v.detach().numpy().astype(np.float32) for k, v in reference.state_dict().items()}
    )
    weights = {"head.layers.0." + k: v for k, v in weights.items()}
    model.weights = weights
    x = np.random.default_rng(2).normal(size=(2, 19, 64)).astype(np.float32)
    mask = np.ones((2, 19), bool)
    mask[1, 6:] = False
    with torch.inference_mode():
        expected = reference(torch.tensor(x), src_key_padding_mask=torch.tensor(~mask)).numpy()
    actual = model._head_layer(x, mask[:, None, None, :], "head.layers.0")
    np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)


def test_numpy_attention_masks_match_reference_semantics():
    valid = np.array([[1, 1, 1, 1, 1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]], bool)
    masks = attention_masks(valid, 4)
    # full: keys = valid positions
    np.testing.assert_array_equal(
        masks["full_attention"][0, 0, 0], [True, True, True, True, True, False, False, False, False, False]
    )
    # local radius 2, inclusive: query 1 sees keys 0-3 (distance <= 2)
    np.testing.assert_array_equal(
        masks["sliding_attention"][0, 0, 1], [True, True, True, True, False, False, False, False, False, False]
    )
    # padded queries can see valid keys so softmax rows stay finite
    np.testing.assert_array_equal(
        masks["sliding_attention"][0, 0, 9], [True, True, True, True, True, False, False, False, False, False]
    )
    np.testing.assert_array_equal(
        masks["sliding_attention"][1, 0, 0], [True, False, False, False, False, False, False, False, False, False]
    )


def test_numpy_decision_model_end_to_end_tiny_random():
    from laya_mlx.numpy_model import NumpyDecisionModel

    cfg = EncoderConfig.from_dict(torch_config().to_dict())
    agent_cfg = {"head_layers": 1, "act_costs": {"escalate": 0.5}}
    model = NumpyDecisionModel(cfg, agent_cfg)
    rng = np.random.default_rng(3)
    model.load_weights(
        {name: (rng.standard_normal(shape) * 0.02).astype(np.float32) for name, shape in model.weight_spec().items()}
    )
    ids = np.array([[2, 5, 6, 7, 3, 0], [2, 8, 9, 3, 0, 0]], np.int32)
    mask = np.array([[1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 0, 0]], bool)
    marker_pos = np.array([[3, 4], [2, 3]], np.int32)
    marker_mask = np.ones((2, 2), bool)
    qtype = np.array([0, 2], np.int32)
    logits, action = model(ids, mask, marker_pos, marker_mask, qtype)
    assert logits.shape == (2, 2)
    assert action.shape == (2, 2)
    assert np.isfinite(logits).all() and np.isfinite(action).all()
    probs = np.exp(logits - logits.max(-1, keepdims=True))
    probs /= probs.sum(-1, keepdims=True)
    assert np.allclose(probs.sum(-1), 1.0)


def test_numpy_load_weights_rejects_missing_and_misshaped():
    cfg = EncoderConfig.from_dict(torch_config().to_dict())
    model = NumpyDecisionModel(cfg, {"head_layers": 1, "act_costs": {}})
    spec = model.weight_spec()
    full = {
        name: (np.random.default_rng(4).standard_normal(shape) * 0.02).astype(np.float32)
        for name, shape in spec.items()
    }
    missing = {k: v for k, v in full.items() if k != "temperature"}
    with pytest.raises(ValueError, match="missing"):
        model.load_weights(missing)
    bad = dict(full)
    bad["temperature"] = np.zeros(2, np.float32)
    with pytest.raises(ValueError, match="shape"):
        model.load_weights(bad)
