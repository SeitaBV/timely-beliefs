"""Continuous ranked probability score (CRPS) for ensemble forecasts.

Vendored (and trimmed to the single function actually used by timely-beliefs)
from properscoring 0.1, Copyright 2015 The Climate Corporation, licensed under
the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0).
Source files (pinned to the v0.1 tag, matching the PyPI 0.1 wheel):
- https://github.com/TheClimateCorporation/properscoring/blob/397f97020ea6c06621f9563fbcecfa31ffccc8d7/properscoring/_crps.py
- https://github.com/TheClimateCorporation/properscoring/blob/397f97020ea6c06621f9563fbcecfa31ffccc8d7/properscoring/_utils.py

Vendored to reduce the number of (transitive) dependencies.
"""

from __future__ import annotations

import contextlib
import warnings

import numpy as np


def _move_axis_to_end(array: np.ndarray, axis: int) -> np.ndarray:
    array = np.asarray(array)
    return np.rollaxis(array, axis, start=array.ndim)


def _argsort_indices(a: np.ndarray, axis: int = -1) -> tuple:
    """Like argsort, but returns an index suitable for sorting the original
    array even if that array is multidimensional."""
    a = np.asarray(a)
    ind = list(np.ix_(*[np.arange(d) for d in a.shape]))
    ind[axis] = a.argsort(axis)
    return tuple(ind)


@contextlib.contextmanager
def _suppress_warnings(msg: str | None = None):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", msg)
        yield


def _crps_ensemble_vectorized(observations, forecasts, weights=1):
    """CRPS via the identity CRPS(F, x) = E_F|X - x| - 1/2 * E_F|X - X'|,
    where X and X' are independent random variables drawn from the forecast
    distribution F. Runtime O(n^2) in the ensemble size, but needs only numpy.
    """
    observations = np.asarray(observations)
    forecasts = np.asarray(forecasts)
    weights = np.asarray(weights)
    if weights.ndim > 0:
        weights = np.where(~np.isnan(forecasts), weights, np.nan)
        weights = weights / np.nanmean(weights, axis=-1, keepdims=True)

    if observations.ndim == forecasts.ndim - 1:
        # sum over the last (ensemble) axis
        assert observations.shape == forecasts.shape[:-1]
        observations = observations[..., np.newaxis]
        with _suppress_warnings("Mean of empty slice"):
            score = np.nanmean(weights * abs(forecasts - observations), -1)
        # insert new axes along last and second to last forecast dimensions so
        # forecasts_diff expands with the array broadcasting
        forecasts_diff = np.expand_dims(forecasts, -1) - np.expand_dims(
            forecasts, -2
        )
        weights_matrix = np.expand_dims(weights, -1) * np.expand_dims(
            weights, -2
        )
        with _suppress_warnings("Mean of empty slice"):
            score += -0.5 * np.nanmean(
                weights_matrix * abs(forecasts_diff), axis=(-2, -1)
            )
        return score
    elif observations.ndim == forecasts.ndim:
        # no 'realization' axis to sum over (this is a deterministic forecast)
        return abs(observations - forecasts)


def crps_ensemble(
    observations, forecasts, weights=None, issorted: bool = False, axis: int = -1
):
    """Calculate the continuous ranked probability score (CRPS) for a set of
    explicit forecast realizations, compared against a scalar observation.

    See properscoring.crps_ensemble for the full docstring; this is a
    pure-numpy vendored copy of that function's non-numba code path.
    """
    observations = np.asarray(observations)
    forecasts = np.asarray(forecasts)
    if axis != -1:
        forecasts = _move_axis_to_end(forecasts, axis)

    if weights is not None:
        weights = _move_axis_to_end(weights, axis)
        if weights.shape != forecasts.shape:
            raise ValueError("forecasts and weights must have the same shape")

    if observations.shape not in [forecasts.shape, forecasts.shape[:-1]]:
        raise ValueError(
            "observations and forecasts must have matching shapes or "
            "matching shapes except along `axis=%s`" % axis
        )

    if observations.shape == forecasts.shape:
        if weights is not None:
            raise ValueError(
                "cannot supply weights unless you also supply an ensemble forecast"
            )
        return abs(observations - forecasts)

    if not issorted:
        if weights is None:
            forecasts = np.sort(forecasts, axis=-1)
        else:
            idx = _argsort_indices(forecasts, axis=-1)
            forecasts = forecasts[idx]
            weights = weights[idx]

    if weights is None:
        weights = np.ones_like(forecasts)

    return _crps_ensemble_vectorized(observations, forecasts, weights)
