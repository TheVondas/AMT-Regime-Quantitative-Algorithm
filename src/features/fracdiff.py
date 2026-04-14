"""
Fractional differencing — López de Prado (2018) fixed-width window method.

Standard integer differencing (d=1) makes a non-stationary series stationary
but destroys long-term memory. Fractional differencing finds the minimum d
(between 0 and 1) that achieves stationarity while preserving maximum memory.

At d=0.3 for example, ~70% of the series memory is retained while removing
enough non-stationarity for ML models to learn from.

Implementation follows Chapter 5 of "Advances in Financial Machine Learning"
(López de Prado, 2018). Uses a fixed-width window to keep computation
tractable and avoid infinite weight series.

No external dependencies — pure numpy/pandas implementation, replacing the
`fracdiff` package which is not compatible with Python 3.14.
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller


def _get_weights(d: float, window: int) -> np.ndarray:
    """Compute fractional differencing weights using recursive formula.

    Weights are derived from the binomial series expansion:
        w_0 = 1
        w_k = -w_{k-1} * (d - k + 1) / k

    Args:
        d: Fractional differencing order (0 < d < 1).
        window: Number of weights to compute (fixed window width).

    Returns:
        Array of weights, shape (window, 1).
    """
    weights = [1.0]
    for k in range(1, window):
        w = -weights[-1] * (d - k + 1) / k
        weights.append(w)
    return np.array(weights[::-1]).reshape(-1, 1)


def frac_diff(series: pd.Series, d: float, window: int = 100) -> pd.Series:
    """Apply fractional differencing to a series.

    Convolves the series with fractional differencing weights over a fixed
    window. The first (window - 1) values will be NaN due to insufficient
    history.

    Args:
        series: Input time series.
        d: Fractional differencing order (0 < d ≤ 1). d=0 returns the
            original series, d=1 is equivalent to first differencing.
        window: Fixed window width for weight computation. Larger windows
            retain more memory but are slower. Default 100 is sufficient
            for daily financial data.

    Returns:
        Fractionally differenced series with same index as input.
    """
    weights = _get_weights(d, window)
    result = pd.Series(index=series.index, dtype=float)

    for i in range(window - 1, len(series)):
        segment = series.iloc[i - window + 1 : i + 1].values.reshape(-1, 1)
        result.iloc[i] = np.dot(weights.T, segment).item()

    return result


def find_min_d(
    series: pd.Series,
    max_d: float = 1.0,
    significance: float = 0.05,
    window: int = 100,
    d_step: float = 0.05,
) -> float:
    """Find the minimum d that makes a series stationary at the given significance.

    Tests d values from 0 upward in steps of d_step. Returns the first d
    where the ADF test rejects the null hypothesis of a unit root.

    If the series is already stationary at d=0, returns 0.0 (no differencing
    needed).

    Args:
        series: Input time series.
        max_d: Maximum d to test (default 1.0).
        significance: ADF p-value threshold (default 0.05).
        window: Fixed window width for fractional differencing.
        d_step: Step size for d search (default 0.05).

    Returns:
        Minimum d value that achieves stationarity. Returns max_d if
        stationarity is not achieved.
    """
    # First check if series is already stationary
    clean = series.dropna()
    if len(clean) < 50:
        return 0.0

    result = adfuller(clean.values, regression="c", autolag="AIC")
    if result[1] < significance:
        return 0.0

    # Search for minimum d
    d = d_step
    while d <= max_d:
        diffed = frac_diff(series, d, window=window).dropna()
        if len(diffed) < 50:
            d += d_step
            continue

        result = adfuller(diffed.values, regression="c", autolag="AIC")
        if result[1] < significance:
            return round(d, 2)

        d += d_step

    return round(max_d, 2)
