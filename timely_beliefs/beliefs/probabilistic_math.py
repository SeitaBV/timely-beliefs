"""Small, dependency-free math helpers shared by both joint-cdf implementations.

Own module so independent_joint_cdf.py and probabilistic_backend.py can both import from it
without a circular import back through probabilistic_utils.py.
"""

from __future__ import annotations

from statistics import NormalDist

import numpy as np


def cp_to_p(cp: list[float] | np.ndarray) -> np.ndarray:
    """Convert numpy array of cumulative probabilities to probabilities. If list, cast to numpy array."""
    return np.concatenate(([cp[0]], np.diff(cp))) if len(cp) != 0 else np.empty(0)


def erfinv(y: float) -> float:
    """Inverse of the error function.

    erf(z) = 2 * Phi(z * sqrt(2)) - 1, so erfinv(y) = Phi^-1((y + 1) / 2) / sqrt(2),
    where Phi is the standard normal cdf (whose inverse is stdlib's NormalDist.inv_cdf).
    """
    return NormalDist().inv_cdf((y + 1) / 2) / 2**0.5
