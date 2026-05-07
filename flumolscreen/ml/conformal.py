"""Conformal-calibration helpers for uncertainty-aware inference."""

from __future__ import annotations

import numpy as np
import pandas as pd

CONFORMAL_EPSILON = 1e-6

__all__ = [
    "CONFORMAL_EPSILON",
    "apply_symmetric_conformal_interval",
    "compute_symmetric_conformal_half_width",
    "compute_absolute_standardized_residuals",
    "fit_symmetric_conformal_scaler",
]


def compute_absolute_standardized_residuals(
    y_true: pd.Series,
    prediction_mean: pd.Series,
    prediction_std: pd.Series,
) -> pd.Series:
    """Compute absolute standardized residuals for symmetric conformal scaling."""
    # Measure absolute calibration error around the ensemble mean prediction.
    absolute_residuals = (y_true - prediction_mean).abs()

    # Guard against zero spread so standardized residuals stay finite.
    safe_prediction_std = prediction_std.clip(lower=CONFORMAL_EPSILON)
    return absolute_residuals / safe_prediction_std


def fit_symmetric_conformal_scaler(
    z_scores: pd.Series,
    interval_coverage: float,
) -> float:
    """Fit the symmetric conformal multiplier for the target coverage."""
    if not 0 < interval_coverage < 1:
        raise ValueError("interval_coverage must be between 0 and 1")
    if z_scores.empty:
        raise ValueError("z_scores must contain at least one value")

    # Use the target upper quantile so interval half-width matches held-out errors.
    return float(np.quantile(z_scores.to_numpy(), interval_coverage))


def apply_symmetric_conformal_interval(
    prediction_mean: pd.Series,
    prediction_std: pd.Series,
    q: float,
) -> pd.DataFrame:
    """Apply a symmetric conformal multiplier to ensemble predictions."""
    # Reuse the calibrated half-width so interval endpoints stay consistent with errors.
    half_width = compute_symmetric_conformal_half_width(
        prediction_std=prediction_std,
        q=q,
    )

    # Return lower and upper endpoints; the half-width is symmetric, not the endpoints.
    return pd.DataFrame(
        {
            "interval_lower": prediction_mean - half_width,
            "interval_upper": prediction_mean + half_width,
        },
        index=prediction_mean.index,
    )


def compute_symmetric_conformal_half_width(
    prediction_std: pd.Series,
    q: float,
) -> pd.Series:
    """Convert ensemble spread into the symmetric conformal error half-width."""
    # Scale the ensemble spread by the calibrated multiplier and guard near-zero spread.
    return q * prediction_std.clip(lower=CONFORMAL_EPSILON)
