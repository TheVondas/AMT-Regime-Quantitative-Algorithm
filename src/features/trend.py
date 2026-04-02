"""
Trend features for regime classification.

Computes: ADX (14-day), +DI/-DI (14-day), Price/SMA(50), Price/SMA(200),
SMA(50)/SMA(200) crossover ratio.

ADX measures trend strength (0-100) regardless of direction.
+DI/-DI measure directional pressure — the AMT concept of initiative vs
responsive activity expressed numerically.
SMA ratios capture trend structure at short and long timeframes.

All features are computed from the daily DataFrame in data/processed/daily.parquet.
"""

import pandas as pd
from ta.trend import ADXIndicator


def compute_adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.DataFrame:
    """Average Directional Index and Directional Indicators.

    ADX measures trend strength (not direction):
      - ADX > 25: strong trend (up or down)
      - ADX < 20: weak trend or ranging market

    +DI measures upward pressure, -DI measures downward pressure:
      - +DI > -DI: buyers more aggressive
      - -DI > +DI: sellers dominating

    Args:
        high: Daily high prices.
        low: Daily low prices.
        close: Daily closing prices.
        period: Lookback window in trading days.

    Returns:
        DataFrame with columns: adx_14, plus_di_14, minus_di_14.
    """
    indicator = ADXIndicator(high=high, low=low, close=close, window=period)
    return pd.DataFrame(
        {
            f"adx_{period}": indicator.adx(),
            f"plus_di_{period}": indicator.adx_pos(),
            f"minus_di_{period}": indicator.adx_neg(),
        }
    )


def compute_price_sma_ratio(close: pd.Series, period: int) -> pd.Series:
    """Price relative to its Simple Moving Average, expressed as a ratio.

    Ratio > 1.0 means price is above its moving average (bullish).
    Ratio < 1.0 means price is below (bearish).
    Using a ratio rather than a difference makes the feature comparable
    across the full price history (SPY $80 in 2005 vs $650 in 2026).

    Args:
        close: Daily closing prices.
        period: SMA lookback window in trading days.

    Returns:
        Ratio of close / SMA(period).
    """
    sma = close.rolling(window=period).mean()
    return close / sma


def compute_sma_cross_ratio(
    close: pd.Series, fast: int = 50, slow: int = 200
) -> pd.Series:
    """SMA crossover ratio — fast SMA relative to slow SMA.

    Ratio > 1.0: golden cross territory (fast above slow, bullish structure).
    Ratio < 1.0: death cross territory (fast below slow, bearish structure).

    Continuous ratio rather than binary flag so the classifier can learn
    how established the crossover is (1.08 is very different from 1.001).

    This is the slowest-moving trend feature — captures major structural
    shifts and acts as an anchor against short-term noise.

    Args:
        close: Daily closing prices.
        fast: Fast SMA period (default 50).
        slow: Slow SMA period (default 200).

    Returns:
        Ratio of SMA(fast) / SMA(slow).
    """
    sma_fast = close.rolling(window=fast).mean()
    sma_slow = close.rolling(window=slow).mean()
    return sma_fast / sma_slow


def build_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build all trend features from the daily DataFrame.

    Args:
        df: Daily DataFrame with 'high', 'low', and 'close' columns.

    Returns:
        DataFrame with trend feature columns, same index as input.
        Columns: adx_14, plus_di_14, minus_di_14, price_sma50_ratio,
                 price_sma200_ratio, sma_cross_ratio.
    """
    close = df["close"]
    high = df["high"]
    low = df["low"]

    features = pd.DataFrame(index=df.index)

    # ADX and Directional Indicators (14-day)
    adx_df = compute_adx(high, low, close, period=14)
    features = pd.concat([features, adx_df], axis=1)

    # Price relative to SMA(50) and SMA(200)
    features["price_sma50_ratio"] = compute_price_sma_ratio(close, period=50)
    features["price_sma200_ratio"] = compute_price_sma_ratio(close, period=200)

    # SMA(50) / SMA(200) crossover ratio
    features["sma_cross_ratio"] = compute_sma_cross_ratio(close)

    return features
