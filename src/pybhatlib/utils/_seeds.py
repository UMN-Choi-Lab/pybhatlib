"""Random seed management for reproducibility."""

from __future__ import annotations

import numpy as np


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility across all backends.

    Parameters
    ----------
    seed : int
        Random seed value.
    """
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
