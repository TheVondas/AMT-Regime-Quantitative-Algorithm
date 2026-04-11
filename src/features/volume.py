"""
Volume features for regime classification.

Computes: OBV rate of change (21-day), volume ratio (vs 20-day average),
MFI (14-day), normalised Force Index (13-day EMA).

These features capture the price-volume relationship from different angles:
  - OBV ROC: is volume flow accelerating or decelerating? (stationary)
  - Volume ratio: participation intensity relative to recent norm
  - MFI: volume-weighted momentum oscillator (RSI with conviction)
  - Force Index (normalised): power behind moves, comparable across time

Together they tell the classifier whether price moves are supported by
volume (genuine trends) or hollow (distribution/accumulation).

All features are computed from the daily DataFrame in data/processed/daily.parquet.
"""

import pandas as pd
from ta.volume import ForceIndexIndicator, MFIIndicator, OnBalanceVolumeIndicator


def compute_obv_roc(close: pd.Series, volume: pd.Series, period: int = 21) -> pd.Series:
    """OBV Rate of Change — is volume flow accelerating or decelerating?

    Raw OBV is cumulative and non-stationary (grows from ~0 to ~17B over
    20 years), which would cause the classifier to learn time-dependent
    thresholds. Instead we compute the percentage change of OBV over a
    rolling window, which is stationary and captures the actual signal:
    whether volume is flowing in (positive ROC) or out (negative ROC).

    Rising OBV ROC + rising price = volume confirms trend (healthy).
    Rising price + declining OBV ROC = distribution signal (hollow move).

    Args:
        close: Daily closing prices.
        volume: Daily trading volume.
        period: Lookback window for rate of change in trading days.

    Returns:
        OBV rate of change as a Series (percentage, stationary).
    """
    obv = OnBalanceVolumeIndicator(close=close, volume=volume).on_balance_volume()
    # Use absolute value in denominator to handle periods where OBV is near zero
    return (obv - obv.shift(period)) / obv.shift(period).abs().clip(lower=1)


def compute_volume_ratio(volume: pd.Series, period: int = 20) -> pd.Series:
    """Volume ratio — current volume relative to its rolling average.

    Ratio = 1.0: normal participation.
    Ratio > 1.5: elevated activity (breakouts, panic, major news).
    Ratio < 0.7: low participation (quiet ranging, holidays).

    No stationarity issues since it's already a ratio.
    Potentially a leading indicator — volume spikes often precede
    the full price move in regime transitions.

    Args:
        volume: Daily trading volume.
        period: Rolling average window in trading days.

    Returns:
        Ratio of current volume to its rolling mean.
    """
    avg_volume = volume.rolling(window=period).mean()
    return volume / avg_volume


def compute_mfi(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Money Flow Index — volume-weighted RSI, scaled 0-100.

    Combines price direction with volume to measure buying/selling pressure.
    MFI > 80: heavy buying with volume (overbought).
    MFI < 20: heavy selling with volume (oversold).

    Unlike RSI, MFI weights moves by volume — a price increase on
    10M shares is a stronger signal than the same increase on 1M shares.

    Price near highs but MFI declining = distribution (up moves on
    declining volume, down moves attracting heavier participation).

    Args:
        high: Daily high prices.
        low: Daily low prices.
        close: Daily closing prices.
        volume: Daily trading volume.
        period: Lookback window in trading days.

    Returns:
        MFI values as a Series (0-100 scale).
    """
    return MFIIndicator(
        high=high, low=low, close=close, volume=volume, window=period
    ).money_flow_index()


def compute_force_index_normalised(
    close: pd.Series, volume: pd.Series, period: int = 13
) -> pd.Series:
    """Normalised Force Index — power behind moves, comparable across time.

    Raw Force Index = (close - prev_close) x volume. This suffers from
    scaling: both price ($80 in 2005 vs $650 in 2026) and volume have
    grown, making raw values ~8x larger in recent years purely due to
    scale, not market behaviour.

    We normalise by dividing by (close x 20-day average volume), producing
    a dimensionless ratio that measures force relative to current market
    scale. This makes the feature comparable across the full 20-year history.

    Positive: buyers in control. Negative: sellers in control.
    Oscillating around zero with declining amplitude: ranging market.

    Args:
        close: Daily closing prices.
        volume: Daily trading volume.
        period: EMA smoothing window in trading days.

    Returns:
        Normalised Force Index as a Series (dimensionless).
    """
    raw_force = ForceIndexIndicator(
        close=close, volume=volume, window=period
    ).force_index()
    # Normalise by close x 20-day average volume
    normaliser = close * volume.rolling(window=20).mean()
    return raw_force / normaliser


def build_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build all volume features from the daily DataFrame.

    Args:
        df: Daily DataFrame with 'high', 'low', 'close', and 'volume' columns.

    Returns:
        DataFrame with volume feature columns, same index as input.
        Columns: obv_roc_21, volume_ratio_20, mfi_14, force_index_norm_13.
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]
    volume = df["volume"]

    features = pd.DataFrame(index=df.index)

    # OBV rate of change (21-day) — stationary, captures volume flow direction
    features["obv_roc_21"] = compute_obv_roc(close, volume, period=21)

    # Volume ratio vs 20-day average
    features["volume_ratio_20"] = compute_volume_ratio(volume, period=20)

    # Money Flow Index (14-day)
    features["mfi_14"] = compute_mfi(high, low, close, volume, period=14)

    # Normalised Force Index (13-day EMA)
    features["force_index_norm_13"] = compute_force_index_normalised(
        close, volume, period=13
    )

    return features
