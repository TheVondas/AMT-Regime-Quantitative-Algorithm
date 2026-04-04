from unittest.mock import patch

import pandas as pd
import pytest

from src.data.download import download_ticker


@pytest.fixture
def mock_yf_data():
    """Returns a standard OHLCV dataframe structure."""
    df = pd.DataFrame(
        {
            "Open": [100.0, 101.0],
            "High": [102.0, 103.0],
            "Low": [99.0, 100.0],
            "Close": [101.0, 102.0],
            "Volume": [1000, 1100],
        },
        index=pd.to_datetime(["2025-01-01", "2025-01-02"]),
    )
    return df


def test_download_ticker_success(mock_yf_data):
    """Ensure ticker download returns a correctly formatted DataFrame."""
    with patch("yfinance.download", return_value=mock_yf_data) as mock_get:
        df = download_ticker("SPY", start="2025-01-01")

        assert not df.empty
        assert df.index.name == "date"
        assert "Close" in df.columns
        mock_get.assert_called_once()


def test_download_ticker_raises_on_empty():
    """Ensure it raises ValueError if yfinance returns nothing."""
    with patch("yfinance.download", return_value=pd.DataFrame()):
        with pytest.raises(ValueError, match="No data returned"):
            download_ticker("INVALID_TICKER")
