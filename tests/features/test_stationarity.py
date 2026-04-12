import numpy as np
import pandas as pd

from src.features.stationarity import compute_rolling_adf


def test_compute_rolling_adf_identifies_mean_reversion():
    """A stationary oscillating series should yield a low p-value."""
    data = np.array([0.1, -0.1] * 25)
    returns = pd.Series(data, index=pd.date_range("2025-01-01", periods=50))

    result = compute_rolling_adf(returns, window=30)
    assert (
        result["adf_pvalue_252"].iloc[-1] < 0.05
    )  # p-value for perfectly oscillating series should be close to 0
    assert result["adf_stat_252"].iloc[-1] < 0  # stat should be negative


def test_compute_rolling_adf_handles_zero_variance():
    """ADF is undefined for a constant series; should remain NaN."""
    flat = pd.Series([0.0] * 50)
    result = compute_rolling_adf(flat, window=30)
    assert pd.isna(result["adf_stat_252"].iloc[-1])
