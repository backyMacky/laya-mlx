"""NumPy CPU engine mirroring laya_mlx.model for platforms without Apple silicon.

Same ModernBERT encoder, decision head, checkpoint format and parameter names as
the MLX runtime (see model.py); compute runs in float32 because NumPy has no
fused kernels and its float16 matmul is neither fast nor accurate enough.
"""

import math

import numpy as np

from .config import EncoderConfig, sanitize_weights

__all__ = [
    "EncoderConfig",
    "NumpyDecisionModel",
    "NumpyModernBert",
    "attention_masks",
    "load_safetensors",
    "sanitize_weights",
]

try:
    from scipy.special import erf as _erf
except ImportError:

    def _erf(x):
        # Abramowitz & Stegun 7.1.26; |error| < 1.5e-7, far below GELU needs.
        x = np.asarray(x, dtype=np.float32)
        sign = np.sign(x)
        t = 1.0 / (1.0 + 0.3275911 * np.abs(x))
        poly = t * (
            0.254829592
            + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429)))
        )
        return sign * (1.0 - poly * np.exp(-np.abs(x) * np.abs(x)))


def gelu(x):
    return x * (1.0 + _erf(x * np.float32(math.sqrt(0.5)))) * 0.5


def relu(x):
    return np.maximum(x, 0)


def layer_norm(x, weight, bias, eps):
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    y = (x - mu) / np.sqrt(var + np.float32(eps))
    return y * weight if bias is None else y * weight + bias


def linear(x, weight, bias):
    y = x @ weight.T
    return y if bias is None else y + bias


def softmax(x, axis=-1):
    z = np.exp(x - x.max(axis=axis, keepdims=True))
    return z / z.sum(axis=axis, keepdims=True)


def rope(x, base):
    """Rotate-half RoPE (MLX mx.fast.rope traditional=False), offset 0, scale 1."""
    _, _, length, dim = x.shape
    inv_freq = base ** (-np.arange(0, dim, 2, dtype=np.float32) / dim)
    angles = np.arange(length, dtype=np.float32)[:, None] * inv_freq[None, :]
    cos, sin = np.cos(angles)[None, None], np.sin(angles)[None, None]
    half = dim // 2
    x1, x2 = x[..., :half], x[..., half:]
    return np.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)


def sdpa(q, k, v, mask):
    """Scaled dot-product attention; boolean mask broadcasts over (b, H, Lq, Lk), True = attend.

    A fully-masked row (no valid keys at all) yields zeros instead of NaN.
    """
    scale = np.float32(q.shape[-1] ** -0.5)
    scores = np.where(mask, (q @ k.transpose(0, 1, 3, 2)) * scale, np.float32(-np.inf))
    row_max = scores.max(axis=-1, keepdims=True)
    z = np.exp(scores - np.where(np.isfinite(row_max), row_max, np.float32(0.0)))
    denom = z.sum(axis=-1, keepdims=True)
    z = np.divide(z, denom, out=np.zeros_like(z), where=denom > 0)
    return z @ v


def attention_masks(attention_mask, window):
    """NumPy mirror of laya_mlx.model.attention_masks; True = attend."""
    valid = attention_mask.astype(bool)
    full = valid[:, None, None, :]
    positions = np.arange(valid.shape[1])
    local = np.abs(positions[:, None] - positions[None, :]) <= window // 2
    local = (local[None, None] | ~valid[:, None, :, None]) & full
    return {"full_attention": full, "sliding_attention": local}


def _split_qkv(qkv):
    return tuple(qkv[:, :, i].transpose(0, 2, 1, 3) for i in range(3))


class NumpyModernBert:
    def __init__(self, cfg: EncoderConfig, prefix="encoder."):
        self.config = cfg
        self.prefix = prefix

    def __call__(self, input_ids, attention_mask, w):
        cfg, p = self.config, self.prefix
        x = w[p + "embeddings.tok_embeddings.weight"][input_ids]
        x = layer_norm(
            x,
            w[p + "embeddings.norm.weight"],
            w.get(p + "embeddings.norm.bias"),
            cfg.norm_eps,
        )
        masks = attention_masks(attention_mask, cfg.local_attention)
        for i, kind in enumerate(cfg.layer_types):
            lp = f"{p}layers.{i}"
            # Layer 0 normalises inside the embeddings only (ModernBERT's first-layer rule).
            h = (
                x
                if i == 0
                else layer_norm(
                    x, w[lp + ".attn_norm.weight"], w.get(lp + ".attn_norm.bias"), cfg.norm_eps
                )
            )
            qkv = linear(h, w[lp + ".attn.Wqkv.weight"], w.get(lp + ".attn.Wqkv.bias"))
            qkv = qkv.reshape(
                input_ids.shape[0], input_ids.shape[1], 3, cfg.num_attention_heads, cfg.head_dim
            )
            q, k, v = _split_qkv(qkv)
            base = cfg.rope_base(kind)
            out = sdpa(rope(q, base), rope(k, base), v, masks[kind])
            out = out.transpose(0, 2, 1, 3).reshape(input_ids.shape[0], input_ids.shape[1], -1)
            x = x + linear(out, w[lp + ".attn.Wo.weight"], w.get(lp + ".attn.Wo.bias"))
            h = layer_norm(x, w[lp + ".mlp_norm.weight"], w.get(lp + ".mlp_norm.bias"), cfg.norm_eps)
            value, gate = np.split(
                linear(h, w[lp + ".mlp.Wi.weight"], w.get(lp + ".mlp.Wi.bias")), 2, axis=-1
            )
            x = x + linear(gelu(value) * gate, w[lp + ".mlp.Wo.weight"], w.get(lp + ".mlp.Wo.bias"))
        return layer_norm(
            x, w[p + "final_norm.weight"], w.get(p + "final_norm.bias"), cfg.norm_eps
        )


class NumpyDecisionModel:
    def __init__(self, encoder_config: EncoderConfig, agent_config: dict):
        self.config = encoder_config
        self.encoder = NumpyModernBert(encoder_config)
        self.head_layers = agent_config.get("head_layers", 2)
        self.act_costs = agent_config.get("act_costs", {})
        self.hidden_size = encoder_config.hidden_size
        self.weights = None

    def weight_spec(self):
        cfg = self.config
        D, inter, V = cfg.hidden_size, cfg.intermediate_size, cfg.vocab_size
        spec = {
            "encoder.embeddings.tok_embeddings.weight": (V, D),
            "encoder.embeddings.norm.weight": (D,),
            "encoder.final_norm.weight": (D,),
            "type_emb.weight": (3, D),
            "temperature": (3,),
        }
        if cfg.norm_bias:
            spec["encoder.embeddings.norm.bias"] = (D,)
            spec["encoder.final_norm.bias"] = (D,)
        for i in range(cfg.num_hidden_layers):
            p = f"encoder.layers.{i}"
            spec[f"{p}.attn.Wqkv.weight"] = (3 * D, D)
            spec[f"{p}.attn.Wo.weight"] = (D, D)
            spec[f"{p}.mlp.Wi.weight"] = (2 * inter, D)
            spec[f"{p}.mlp.Wo.weight"] = (D, inter)
            spec[f"{p}.mlp_norm.weight"] = (D,)
            if i > 0:  # layer 0 has no attn_norm; the embeddings norm stands in for it
                spec[f"{p}.attn_norm.weight"] = (D,)
            if cfg.attention_bias:
                spec[f"{p}.attn.Wqkv.bias"] = (3 * D,)
                spec[f"{p}.attn.Wo.bias"] = (D,)
            if cfg.mlp_bias:
                spec[f"{p}.mlp.Wi.bias"] = (2 * inter,)
                spec[f"{p}.mlp.Wo.bias"] = (D,)
            if cfg.norm_bias:
                spec[f"{p}.mlp_norm.bias"] = (D,)
                if i > 0:
                    spec[f"{p}.attn_norm.bias"] = (D,)
        dims = cfg.hidden_size
        for j in range(self.head_layers):
            p = f"head.layers.{j}"
            spec[f"{p}.self_attn.in_proj.weight"] = (3 * dims, dims)
            spec[f"{p}.self_attn.in_proj.bias"] = (3 * dims,)
            spec[f"{p}.self_attn.out_proj.weight"] = (dims, dims)
            spec[f"{p}.self_attn.out_proj.bias"] = (dims,)
            spec[f"{p}.norm1.weight"] = (dims,)
            spec[f"{p}.norm1.bias"] = (dims,)
            spec[f"{p}.norm2.weight"] = (dims,)
            spec[f"{p}.norm2.bias"] = (dims,)
            spec[f"{p}.linear1.weight"] = (4 * dims, dims)
            spec[f"{p}.linear1.bias"] = (4 * dims,)
            spec[f"{p}.linear2.weight"] = (dims, 4 * dims)
            spec[f"{p}.linear2.bias"] = (dims,)
        spec["scorer.layers.0.weight"] = (dims,)
        spec["scorer.layers.0.bias"] = (dims,)
        spec["scorer.layers.1.weight"] = (dims, dims)
        spec["scorer.layers.1.bias"] = (dims,)
        spec["scorer.layers.3.weight"] = (1, dims)
        spec["scorer.layers.3.bias"] = (1,)
        spec["act_head.layers.0.weight"] = (256, dims + 4)
        spec["act_head.layers.0.bias"] = (256,)
        spec["act_head.layers.2.weight"] = (len(self.act_costs) + 1, 256)
        spec["act_head.layers.2.bias"] = (len(self.act_costs) + 1,)
        return spec

    def load_weights(self, weights, dtype=np.float32):
        spec = self.weight_spec()
        missing = sorted(set(spec) - set(weights))
        if missing:
            raise ValueError(f"Checkpoint is missing {len(missing)} parameter(s): {missing[:4]}")
        unexpected = sorted(set(weights) - set(spec))
        if unexpected:
            raise ValueError(f"Checkpoint has {len(unexpected)} unexpected parameter(s): {unexpected[:4]}")
        self.weights = {}
        for name, shape in spec.items():
            value = np.asarray(weights[name])
            if tuple(value.shape) != shape:
                raise ValueError(f"Parameter {name} has shape {value.shape}, expected {shape}")
            self.weights[name] = value.astype(dtype, copy=False)

    def _head_layer(self, x, mask, p):
        dims = x.shape[-1]
        h = layer_norm(x, self.weights[p + ".norm1.weight"], self.weights[p + ".norm1.bias"], 1e-5)
        b, length = x.shape[:2]
        qkv = linear(h, self.weights[p + ".self_attn.in_proj.weight"], self.weights[p + ".self_attn.in_proj.bias"])
        qkv = qkv.reshape(b, length, 3, max(1, dims // 64), dims // max(1, dims // 64))
        q, k, v = _split_qkv(qkv)
        out = sdpa(q, k, v, mask)
        out = linear(
            out.transpose(0, 2, 1, 3).reshape(b, length, dims),
            self.weights[p + ".self_attn.out_proj.weight"],
            self.weights[p + ".self_attn.out_proj.bias"],
        )
        x = x + out
        h = layer_norm(x, self.weights[p + ".norm2.weight"], self.weights[p + ".norm2.bias"], 1e-5)
        # PyTorch TransformerEncoderLayer defaults to ReLU, even though the
        # encoder and scoring heads use GELU.
        h = relu(linear(h, self.weights[p + ".linear1.weight"], self.weights[p + ".linear1.bias"]))
        return x + linear(h, self.weights[p + ".linear2.weight"], self.weights[p + ".linear2.bias"])

    def __call__(self, input_ids, attention_mask, marker_pos, marker_mask, qtype):
        w = self.weights
        h = self.encoder(input_ids, attention_mask, w)
        h = h + w["type_emb.weight"][qtype][:, None, :]
        mask = attention_mask[:, None, None, :]
        for j in range(self.head_layers):
            h = self._head_layer(h, mask, f"head.layers.{j}")
        markers = h[np.arange(h.shape[0])[:, None], np.maximum(marker_pos, 0)]
        logits = layer_norm(markers, w["scorer.layers.0.weight"], w["scorer.layers.0.bias"], 1e-5)
        logits = gelu(linear(logits, w["scorer.layers.1.weight"], w["scorer.layers.1.bias"]))
        logits = linear(logits, w["scorer.layers.3.weight"], w["scorer.layers.3.bias"])
        logits = logits.squeeze(-1).astype(np.float32)
        logits = np.where(marker_mask, logits, np.float32(-1e4))
        p = softmax(logits, axis=-1)
        k = np.maximum(marker_mask.sum(axis=-1), 2).astype(np.float32)
        entropy = -(p * np.log(np.maximum(p, 1e-9))).sum(axis=-1) / np.log(k)
        # The public runtime pads to at least two marker slots for one-option choices.
        top = np.sort(p, axis=-1)[:, -2:]
        features = np.stack([top[:, 1], top[:, 1] - top[:, 0], entropy, k / 255.0], axis=-1)
        pooled = np.concatenate([h[:, 0].astype(np.float32), features], axis=-1)
        action = linear(pooled, w["act_head.layers.0.weight"], w["act_head.layers.0.bias"])
        action = linear(gelu(action), w["act_head.layers.2.weight"], w["act_head.layers.2.bias"])
        return logits, action.astype(np.float32)


def load_safetensors(path):
    from safetensors.numpy import load_file

    return load_file(str(path))
