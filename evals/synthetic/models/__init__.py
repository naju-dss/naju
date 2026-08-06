"""Model registry for the synthetic suite. build_model(name, cfg, vocab, classes)."""
from models import fla_baselines, naju, transformer

_BUILDERS = {
    "naju": naju.build,
    "transformer": transformer.build,
    # flash-linear-attention / mamba-ssm / xlstm wrappers (lazy imports)
    "mamba": fla_baselines.build,
    "mamba2": fla_baselines.build,
    "xlstm": fla_baselines.build,
    "gla": fla_baselines.build,
    "hgrn": fla_baselines.build,
    "rwkv": fla_baselines.build,
    "retnet": fla_baselines.build,
}

MODEL_NAMES = list(_BUILDERS.keys())


def build_model(name, cfg, vocab_size, num_classes):
    if name not in _BUILDERS:
        raise ValueError(f"unknown model {name!r}; choose from {MODEL_NAMES}")
    return _BUILDERS[name](cfg, vocab_size, num_classes)
