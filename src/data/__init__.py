# src/data/__init__.py
"""Data package for dataset ingestion, profile-disjoint partitioning, and split manifests.

`splits` imports numpy at module scope, so re-exporting it eagerly here made
`import src.data.anything` require numpy -- including `dataset_pin`, which
exists partly to produce a helpful error message on a machine that has not
installed anything yet.

Same PEP 562 treatment as `src/rewriter/__init__.py` (PR #44). The public API
is unchanged: `from src.data import create_profile_disjoint_splits` still
works and still imports numpy, just not until something asks for it.
"""

from typing import Any

_LAZY = frozenset({"create_profile_disjoint_splits", "load_or_fetch_synthpai"})

__all__ = ["create_profile_disjoint_splits", "load_or_fetch_synthpai"]


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        from src.data import splits

        return getattr(splits, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
