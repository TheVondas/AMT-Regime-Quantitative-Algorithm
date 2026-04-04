from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.data.clean import build_daily, load_raw


@pytest.fixture
def spy_raw():
    return pd.DataFrame(
        {"close": [100.0, 101.0, 102.0, 103.0], "volume": [10, 20, 30, 40]},
        index=pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04"]),
    )


@pytest.fixture
def vix_raw():
    return pd.DataFrame(
        {"close": [20.0, np.nan, 21.0, 22.0]},  # np.nan on Jan 2nd to test ffill logic
        index=pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04"]),
    )


def test_load_raw_lowercases_columns(tmp_path):
    """Ensure load_raw forces column names to lowercase."""
    df = pd.DataFrame({"CLOSE": [1], "Volume": [2]})
    file_path = tmp_path / "test.parquet"
    df.to_parquet(file_path)

    with patch("src.data.clean.RAW_DIR", tmp_path):
        result = load_raw("test.parquet")
        assert list(result.columns) == ["close", "volume"]


def test_build_daily_alignment_and_log_returns(spy_raw, vix_raw):
    """Verify join logic, forward fill, and log return calculation."""

    def side_effect(filename):
        if "spy" in filename:
            return spy_raw.copy()
        return vix_raw.copy()

    with patch("src.data.clean.load_raw", side_effect=side_effect):
        df = build_daily()

        assert "log_return" in df.columns
        assert df.loc["2025-01-02", "vix"] == 20.0
        assert df.index.min() == pd.to_datetime("2025-01-02")
