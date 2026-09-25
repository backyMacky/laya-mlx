"""Encoder configuration and checkpoint name sanitisation shared by both backends."""

from dataclasses import dataclass, fields


@dataclass
class EncoderConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    model_type: str = "modernbert"
    norm_eps: float = 1e-5
    norm_bias: bool = False
    attention_bias: bool = False
    mlp_bias: bool = False
    hidden_activation: str = "gelu"
    local_attention: int = 128
    global_attn_every_n_layers: int = 3
    global_rope_theta: float = 160000.0
    local_rope_theta: float = 10000.0
    max_position_embeddings: int = 8192
    layer_types: list[str] | None = None
    rope_parameters: dict | None = None

    @classmethod
    def from_dict(cls, value: dict):
        names = {f.name for f in fields(cls)}
        cfg = cls(**{k: v for k, v in value.items() if k in names})
        if cfg.model_type != "modernbert":
            raise ValueError(f"Unsupported encoder: {cfg.model_type!r}; expected modernbert")
        if cfg.hidden_activation != "gelu":
            raise ValueError(f"Unsupported encoder activation: {cfg.hidden_activation!r}")
        if cfg.hidden_size % cfg.num_attention_heads or cfg.head_dim % 2:
            raise ValueError("ModernBERT requires an even, integral attention head dimension")
        if cfg.layer_types is None:
            cfg.layer_types = [
                "full_attention" if i % cfg.global_attn_every_n_layers == 0 else "sliding_attention"
                for i in range(cfg.num_hidden_layers)
            ]
        if len(cfg.layer_types) != cfg.num_hidden_layers or set(cfg.layer_types) - {
            "full_attention",
            "sliding_attention",
        }:
            raise ValueError("Invalid ModernBERT layer_types")
        for kind in set(cfg.layer_types):
            params = (cfg.rope_parameters or {}).get(kind, {})
            if params.get("rope_type", "default") != "default":
                raise ValueError("Only default (unscaled) ModernBERT RoPE is supported")
        return cfg

    @property
    def head_dim(self):
        return self.hidden_size // self.num_attention_heads

    def rope_base(self, kind):
        fallback = self.global_rope_theta if kind == "full_attention" else self.local_rope_theta
        return float((self.rope_parameters or {}).get(kind, {}).get("rope_theta", fallback))


def sanitize_weights(weights):
    """Map upstream PyTorch parameter names to MLX; Linear layouts already match."""
    result = {}
    for name, value in weights.items():
        name = name.replace(".in_proj_weight", ".in_proj.weight")
        name = name.replace(".in_proj_bias", ".in_proj.bias")
        for prefix in ("scorer", "act_head"):
            if name.startswith(prefix + ".") and not name.startswith(prefix + ".layers."):
                name = prefix + ".layers." + name[len(prefix) + 1 :]
        if name in result:
            raise ValueError(f"Duplicate checkpoint parameter after conversion: {name}")
        result[name] = value
    return result
