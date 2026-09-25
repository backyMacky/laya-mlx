import json
import os

import numpy as np
import pytest
from tokenizers import Tokenizer, models, pre_tokenizers

try:
    import mlx.core as mx
    from mlx.utils import tree_flatten

    HAVE_MLX = True
except ImportError:
    mx = None
    HAVE_MLX = False

from laya_mlx.config import EncoderConfig

if HAVE_MLX and os.environ.get("LAYA_MLX_TEST_DEVICE") == "cpu":
    mx.set_default_device(mx.cpu)

TINY_CFG = {
    "model_type": "modernbert",
    "vocab_size": 128,
    "hidden_size": 64,
    "intermediate_size": 96,
    "num_hidden_layers": 3,
    "num_attention_heads": 1,
    "local_attention": 16,
    "max_position_embeddings": 256,
}

TINY_AGENT_CFG = {
    "encoder": "test/tiny",
    "head_layers": 1,
    "max_len": 128,
    "head_max_len": 32,
    "act_costs": {"escalate": 0.5},
    "temperature": [1.3, 1.1, 2.0],
    "temperature_by_options": {"choice:2": 1.7},
}


def _write_tiny_checkpoint(path, weights):
    (path / "encoder").mkdir(parents=True)
    (path / "tokenizer").mkdir()
    (path / "encoder/config.json").write_text(json.dumps(TINY_CFG))
    (path / "rl_agent_config.json").write_text(json.dumps(TINY_AGENT_CFG))
    vocab = {t: i for i, t in enumerate(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "hello"])}
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.save(str(path / "tokenizer/tokenizer.json"))
    (path / "tokenizer/tokenizer_config.json").write_text(
        json.dumps(
            {
                "pad_token": "[PAD]",
                "cls_token": "[CLS]",
                "sep_token": "[SEP]",
                "mask_token": "[MASK]",
            }
        )
    )
    if HAVE_MLX and weights == "mlx":
        from safetensors.numpy import save_file

        from laya_mlx.model import DecisionModel

        mx.random.seed(7)
        model = DecisionModel(EncoderConfig.from_dict(TINY_CFG), TINY_AGENT_CFG)
        save_file(
            {k: np.ascontiguousarray(np.asarray(v)) for k, v in tree_flatten(model.parameters())},
            str(path / "model.safetensors"),
        )
    else:
        from safetensors.numpy import save_file

        from laya_mlx.numpy_model import NumpyDecisionModel

        rng = np.random.default_rng(7)
        spec = NumpyDecisionModel(EncoderConfig.from_dict(TINY_CFG), TINY_AGENT_CFG).weight_spec()
        save_file(
            {
                name: np.ascontiguousarray((rng.standard_normal(shape) * 0.02).astype(np.float16))
                for name, shape in spec.items()
            },
            str(path / "model.safetensors"),
        )
    return path


@pytest.fixture
def tiny_checkpoint(tmp_path):
    if not HAVE_MLX:
        pytest.skip("tiny_checkpoint requires the MLX backend")
    return _write_tiny_checkpoint(tmp_path / "checkpoint", "mlx")


@pytest.fixture
def tiny_checkpoint_numpy(tmp_path):
    return _write_tiny_checkpoint(tmp_path / "checkpoint", "numpy")


@pytest.fixture
def questions():
    return {
        "topic": {"type": "choice", "instructions": "Choose", "criteria": ["a", "b", "c"]},
        "level": {"type": "score", "instructions": "Level", "criteria": ["low", "high"]},
        "yes": {"type": "noul", "instructions": "Is this true?"},
    }
