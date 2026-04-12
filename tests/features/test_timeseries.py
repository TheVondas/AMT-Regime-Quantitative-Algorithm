import numpy as np
import pandas as pd
import pytest

from src.features.timeseries import (
    build_timeseries_features,
    compute_time_reversal_asymmetry,
)


@pytest.fixture
def asymmetric_data():
    """Series with a clear breakout pattern to induce asymmetry."""
    data = np.array([0.01, 0.01, 0.01, 0.05, 0.10] * 60)
    return pd.Series(data, index=pd.date_range("2025-01-01", periods=300))


def test_compute_tra_returns_series_with_correct_window(asymmetric_data):
    """The first 251 rows should be NaN for a 252-day window + 1-day lag."""
    lag = 1
    window = 252
    result = compute_time_reversal_asymmetry(asymmetric_data, lag=lag, window=window)

    assert pd.isna(result.iloc[251])
    assert not pd.isna(result.iloc[252])


def test_compute_tra_on_constant_is_zero():
    """A perfectly flat series should have zero asymmetry."""
    flat = pd.Series([0.02] * 300)
    result = compute_time_reversal_asymmetry(flat, lag=1, window=252)
    assert result.iloc[-1] == 0.0


def test_build_timeseries_features_columns(asymmetric_data):
    """All three lags are present in the output."""
    df = pd.DataFrame({"log_return": asymmetric_data})
    features = build_timeseries_features(df)
    expected = ["tra_lag1_252", "tra_lag2_252", "tra_lag3_252"]
    assert all(col in features.columns for col in expected)
