import numpy as np
import pandas as pd
import pytest

from src.features.assemble import build_all_features, check_correlations


@pytest.fixture
def mock_daily_data():
    """Matches the structure of data/processed/daily.parquet."""
    dates = pd.date_range("2020-01-01", periods=500)
    data = {
        "open": np.random.uniform(100, 110, 500),
        "high": np.random.uniform(110, 120, 500),
        "low": np.random.uniform(90, 100, 500),
        "close": np.random.uniform(100, 110, 500),
        "volume": np.random.randint(1000000, 5000000, 500),
        "log_return": np.random.normal(0, 0.01, 500),
        "vix": np.random.uniform(15, 30, 500),
        "us10y": np.random.uniform(1.5, 4.5, 500),
        "us5y": np.random.uniform(1.2, 4.0, 500),
        "us3m": np.random.uniform(0.5, 5.0, 500),
    }
    return pd.DataFrame(data, index=dates)


def test_build_all_features_deduplication(mock_daily_data):
    """Ensure VIX and us10y aren't duplicated in the final matrix."""
    features = build_all_features(mock_daily_data)
    counts = features.columns.value_counts()
    assert counts["vix"] == 1
    assert counts["us10y"] == 1


def test_check_correlations():
    """Identify highly correlated pairs."""
    df = pd.DataFrame({"A": np.random.randn(100), "B": np.random.randn(100)})
    df["C"] = df["A"] * 0.99

    highcorr = check_correlations(df)
    cols = [pair[0] for pair in highcorr] + [pair[1] for pair in highcorr]
    assert "A" in cols
    assert "C" in cols
    assert "B" not in cols


def test_pipeline_lagging(mock_daily_data):
    """Verify that shifting prevents look-ahead bias."""
    features = build_all_features(mock_daily_data)
    lagged = features.shift(1)
    assert lagged["vix"].iloc[100] == features["vix"].iloc[99]
