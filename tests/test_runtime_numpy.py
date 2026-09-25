"""Runtime tests for the NumPy CPU backend (mirror of tests/test_runtime.py core cases)."""

import subprocess
import sys

import numpy as np
import pytest

from laya_mlx import Agent
from laya_mlx.agent import collate_items
from laya_mlx.common import render_options


def test_numpy_backend_does_not_import_torch_transformers_or_mlx():
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import laya_mlx, sys; "
                "assert 'torch' not in sys.modules; "
                "assert 'transformers' not in sys.modules; "
                "assert 'mlx.core' not in sys.modules"
            ),
        ],
        check=True,
    )


def test_numpy_all_primitives_empty_request_and_chunking(
    tiny_checkpoint_numpy, questions
):
    agent = Agent(tiny_checkpoint_numpy, dtype="float16", batch_size=16)
    together = agent.predict({"text": "hello"}, questions)
    agent.batch_size = 1
    separate = agent.predict({"text": "hello"}, questions)
    assert together == separate
    assert set(together["answers"]) == set(questions)
    assert together["usage"]["output_tokens"] == 0
    assert 0 <= together["answers"]["yes"]["noul"] <= 1
    assert 0 <= together["answers"]["level"]["score"] <= 1
    assert agent.predict("", {})["answers"] == {}
    assert agent.predict("", {})["usage"]["input_tokens"] == 0


def test_numpy_deterministic_across_instances(tiny_checkpoint_numpy, questions):
    first = Agent(tiny_checkpoint_numpy).predict("hello", questions)
    second = Agent(tiny_checkpoint_numpy).predict("hello", questions)
    assert first == second


def test_numpy_single_option_and_long_state(tiny_checkpoint_numpy):
    agent = Agent(tiny_checkpoint_numpy)
    result = agent.predict(
        "hello " * 1000, {"one": {"type": "choice", "instructions": "choose", "criteria": ["only"]}}
    )
    assert result["answers"]["one"]["probabilities"] == {"only": 1.0}
    assert result["usage"]["input_tokens"] == 128


def test_numpy_structured_criteria_and_mask_injection(tiny_checkpoint_numpy):
    q = Agent._to_internal(
        {
            "type": "noul",
            "instructions": {"task": "verify"},
            "criteria": {"false": {"reason": "no"}, "true": {"reason": "yes"}},
        }
    )
    assert render_options(q) == ['false: {"reason": "no"}', 'true: {"reason": "yes"}']
    agent = Agent(tiny_checkpoint_numpy)
    items, _ = agent.prepare(
        "[MASK] hello [MASK]", {"x": {"type": "noul", "instructions": "[MASK] true?"}}
    )
    assert items[0]["ids"].count(agent.tok.mask_token_id) == 2


def test_numpy_collation_never_marks_padding_as_an_option():
    batch = collate_items(
        [
            {"ids": [1, 2, 3], "markers": [1, 2], "qtype": 0},
            {"ids": [1, 2], "markers": [1], "qtype": 1},
        ],
        0,
    )
    np.testing.assert_array_equal(batch["marker_mask"], [[True, True], [True, False]])
    np.testing.assert_array_equal(
        batch["attention_mask"], [[True, True, True], [True, True, False]]
    )


def test_numpy_cached_prefixes_preserve_inputs(tiny_checkpoint_numpy, questions):
    original = Agent(tiny_checkpoint_numpy)
    cached = Agent(tiny_checkpoint_numpy, cache_prompts=True)
    cached._prefix_cache.capacity = 3
    states = ["", "[MASK] hello", "hello " * 1000, {"text": "你好", "flag": False}]
    for state in states:
        for count in (2, 5, 12):
            questions["topic"]["criteria"] = {str(i): {"value": i} for i in range(count)}
            assert original.prepare(state, questions) == cached.prepare(state, questions)
            assert len(cached._prefix_cache.entries) <= 3
    assert cached.prepare("", {}) == ([], [])


def test_numpy_device_error_message(tiny_checkpoint_numpy):
    with pytest.raises(RuntimeError, match="Apple silicon"):
        Agent(tiny_checkpoint_numpy, device="gpu")


def test_numpy_shortlist_embed_fn(tiny_checkpoint_numpy):
    from laya_mlx.shortlist import embed_fn_from_agent

    agent = Agent(tiny_checkpoint_numpy)
    embed_fn = embed_fn_from_agent(agent)
    vectors = embed_fn(["hello", "", None])
    assert vectors.shape == (3, 64)
    assert np.isfinite(vectors).all()
    assert not np.allclose(vectors[0], vectors[1])
