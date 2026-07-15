"""Reproducibility helpers for deterministic Phase 2 experiments."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np


DEFAULT_RANDOM_SEED = 42


@dataclass(frozen=True)
class ReproducibilityContext:
    seed: int
    python_hash_seed: str


def set_global_seed(seed: int = DEFAULT_RANDOM_SEED) -> ReproducibilityContext:
    """Set common random seeds used by local preprocessing and baselines."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    return ReproducibilityContext(seed=seed, python_hash_seed=os.environ["PYTHONHASHSEED"])

