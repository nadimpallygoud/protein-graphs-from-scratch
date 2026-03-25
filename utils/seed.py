"""Random seed helpers used across scripts."""

from __future__ import annotations

import random

import numpy as np


DEFAULT_SEED = 42


def set_seed(seed: int = DEFAULT_SEED) -> None:
    """Keep experiments reproducible across Python, NumPy, and Torch when available."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

