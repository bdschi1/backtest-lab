"""Bridges to existing Tier 1 repos — reuse what's already built."""

from __future__ import annotations

import os
from pathlib import Path


def resolve_tier1_repo(repo_name: str) -> Path:
    """Resolve path to a sibling repo under the Tier_1 directory.

    Uses TIER1_ROOT env var if set, otherwise walks up from this file
    to find the Tier_1 parent directory.
    """
    root = os.environ.get("TIER1_ROOT")
    if root:
        return Path(root) / repo_name
    for parent in Path(__file__).resolve().parents:
        if parent.name == "Tier_1":
            return parent / repo_name
    return Path(repo_name)


def resolve_aah() -> Path:
    """Resolve path to AAH repo (sibling of Tier_1 under bds_repos).

    Uses AAH_ROOT env var if set, otherwise derives from Tier_1 location.
    """
    root = os.environ.get("AAH_ROOT")
    if root:
        return Path(root)
    for parent in Path(__file__).resolve().parents:
        if parent.name == "Tier_1":
            return parent.parent / "aah"
    return Path("aah")
