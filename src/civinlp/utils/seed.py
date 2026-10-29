"""
Global seed utility for reproducibility.

Sets seeds for Python's random, NumPy, and PyTorch to ensure
deterministic behavior across runs.
"""

import logging
import random

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_SEED = 42


def set_global_seed(seed: int = DEFAULT_SEED) -> None:
    """
    Set the global random seed for reproducibility.

    Args:
        seed: Integer seed value. Defaults to 42.
    """
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        logger.info(f"Set global seed to {seed} (random, numpy, torch)")
    except ImportError:
        logger.info(f"Set global seed to {seed} (random, numpy; torch not available)")
