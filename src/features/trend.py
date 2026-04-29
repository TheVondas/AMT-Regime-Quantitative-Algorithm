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
        high (pd.Series): Daily high prices.
        low (pd.Series): Daily low prices.
        close (pd.Series): Daily closing prices.
        period (pd.Series): Lookback window in trading days (default 14).

    Returns:
        pd.DataFrame: With columns
            "adx_{period}", "plus_di_{period}", "minus_di_{period}".
    """
    assert period > 0, f"adx period ({period}) must be a positive integer."
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
        close (pd.Series): Daily closing prices.
        period (int): SMA lookback window in trading days.

    Returns:
        pd.Series: Ratio of close / SMA(period).
    """
    assert period > 0, f"SMA period ({period}) must be a positive integer."
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
        close (pd.Series): Daily closing prices.
        fast (int): Fast SMA period (default 50).
        slow (int): Slow SMA period (default 200).

    Returns:
        pd.Series: Ratio of SMA(fast) / SMA(slow).
    """
    assert 0 < fast < slow, (
        f"fast SMA period ({fast}) must be a positive "
        f"integer less than slow SMA period ({slow})."
    )
    sma_fast = close.rolling(window=fast).mean()
    sma_slow = close.rolling(window=slow).mean()
    return sma_fast / sma_slow


def build_trend_features(
    df: pd.DataFrame, adx_period: int = 14, sma_fast: int = 50, sma_slow: int = 200
) -> pd.DataFrame:
    """Build all trend features from the daily DataFrame.

    Args:
        df (pd.DataFrame): Daily DataFrame with 'high', 'low', and 'close' columns.
        adx_period (int): Lookback window in trading days
            for adx computation (default 14).
        sma_fast (int): Fast SMA period for ratio and crossover (default 50).
        sma_slow (int): Slow SMA period for ratio and crossover (default 200).

    Returns:
        pd.DataFrame: Trend features with dynamic column names:
            - adx_{adx_period}, plus_di_{adx_period}, minus_di_{adx_period}
            - price_sma{sma_fast}_ratio
            - price_sma{sma_slow}_ratio
            - sma_cross_{sma_fast}_{sma_slow}_ratio
    """
    assert 0 < sma_fast < sma_slow, (
        f"fast SMA period ({sma_fast}) must be a positive "
        f"integer less than slow SMA period ({sma_slow})."
    )
    close = df["close"]
    high = df["high"]
    low = df["low"]

    features = pd.DataFrame(index=df.index)

    # ADX and Directional Indicators
    adx_df = compute_adx(high, low, close, period=adx_period)
    features = pd.concat([features, adx_df], axis=1)

    # Price relative to SMA(sma_period_fast) and SMA(sma_period_slow)
    sma_fast_ratio = compute_price_sma_ratio(close, period=sma_fast)
    sma_slow_ratio = compute_price_sma_ratio(close, period=sma_slow)

    features[f"price_sma{sma_fast}_ratio"] = sma_fast_ratio
    features[f"price_sma{sma_slow}_ratio"] = sma_slow_ratio

    # SMA(sma_period_fast) / SMA(sma_period_slow) crossover ratio
    features["sma_cross_ratio"] = sma_slow_ratio / sma_fast_ratio

    return features
