# src/data/__init__.py
"""Data package for dataset ingestion, profile-disjoint partitioning, and split manifests."""

from src.data.splits import create_profile_disjoint_splits, load_or_fetch_synthpai

__all__ = ["create_profile_disjoint_splits", "load_or_fetch_synthpai"]
