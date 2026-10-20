"""InferenceGuard Rewriter Module.

Heavy submodules (inference and generate_training_data) are resolved on first
access via PEP 562 __getattr__ to avoid circular imports during -m execution
and prevent eager torch initialization.
"""

from typing import Any

_LAZY_INFERENCE = frozenset({"DEFAULT_BASE_MODEL", "QwenRewriterInference", "clean_gpu_memory"})
_LAZY_GEN = frozenset({"DEFAULT_INSTRUCTION", "SYSTEM_PROMPT", "generate_candidate_rewrites_heuristic"})

__all__ = [
    "DEFAULT_BASE_MODEL",
    "DEFAULT_INSTRUCTION",
    "QwenRewriterInference",
    "SYSTEM_PROMPT",
    "clean_gpu_memory",
    "generate_candidate_rewrites_heuristic",
]


def __getattr__(name: str) -> Any:
    """Import the requested submodule attributes lazily when accessed."""
    if name in _LAZY_INFERENCE:
        from src.rewriter import inference

        return getattr(inference, name)
    if name in _LAZY_GEN:
        from src.rewriter import generate_training_data

        return getattr(generate_training_data, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
