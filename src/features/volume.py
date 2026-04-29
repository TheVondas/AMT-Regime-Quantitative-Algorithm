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

from typing import Callable, List

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
        close (pd.Series): Daily closing prices.
        volume (pd.Series): Daily trading volume.
        period (int): Lookback window for rate of change in trading days (default 21).

    Returns:
        pd.Series: OBV rate of change (percentage, stationary).
    """
    assert period > 0, f"period ({period}) must be a positive integer."
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
        volume (pd.Series): Daily trading volume.
        period (int): Rolling average window in trading days (default 20).

    Returns:
        pd.Series: Ratio of current volume to its rolling mean.
    """
    assert period > 0, f"period ({period}) must be a positive integer."
    avg_volume = volume.rolling(window=period).mean()
    return volume / avg_volume


def compute_mfi(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Money Flow Index (MFI) — volume-weighted RSI, scaled 0-100.

    Combines price direction with volume to measure buying/selling pressure.
    MFI > 80: heavy buying with volume (overbought).
    MFI < 20: heavy selling with volume (oversold).

    Unlike RSI, MFI weights moves by volume — a price increase on
    10M shares is a stronger signal than the same increase on 1M shares.

    Price near highs but MFI declining = distribution (up moves on
    declining volume, down moves attracting heavier participation).

    Args:
        high (pd.Series): Daily high prices.
        low (pd.Series): Daily low prices.
        close (pd.Series): Daily closing prices.
        volume (pd.Series): Daily trading volume.
        period (int): Lookback window in trading days (default 14).

    Returns:
        pd.Series: MFI values as a Series (0-100 scale).
    """
    assert period > 0, f"period ({period}) must be a positive integer."
    return MFIIndicator(
        high=high, low=low, close=close, volume=volume, window=period
    ).money_flow_index()


def compute_force_index_normalised(
    close: pd.Series, volume: pd.Series, period: int = 13, norm_window: int = 20
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
        close (pd.Series): Daily closing prices.
        volume (pd.Series): Daily trading volume.
        period (int): EMA smoothing window in trading days.

    Returns:
        pd.Series: Normalised Force Index as a Series (dimensionless).
    """
    assert period > 0, f"Smoothing period ({period}) must be positive."
    assert norm_window > 0, f"Normalisation window ({norm_window}) must be positive."
    raw_force = ForceIndexIndicator(
        close=close, volume=volume, window=period
    ).force_index()

    normaliser = close * volume.rolling(window=norm_window).mean()
    return raw_force / normaliser


def _compute_for_periods(
    period: List[int] | int,
    compute_func: Callable[[int], pd.Series],
    feat_name: str,
    suffix: str = "",
    index: pd.Index = None,
) -> pd.DataFrame:
    """Helper to map a compute function over multiple lookback periods.

    Args:
        periods (List[int] | int): Lookback window(s).
        compute_func (Callable): Function taking (int period) -> Series.
        feat_name (str): Prefix for the column name.
        suffix (str): Suffix to append after the period (e.g., '_pct').
        index (pd.Index): Index for the resulting DataFrame.

    Returns:
        pd.DataFrame: DataFrame with columns named '{feat_name}_{p}{suffix}'.
    """
    if isinstance(period, int):
        assert period > 0, f"{feat_name}_period ({period}) must be a positive integer."
        period = [period]

    computations = pd.DataFrame(index=index)
    visited = set()

    for p in period:
        if p in visited:
            continue
        assert p > 0, f"Period must be positive. Found {p} for {feat_name}_period."
        col_name = f"{feat_name}_{p}{suffix}"
        computations[col_name] = compute_func(p)
        visited.add(p)

    return computations


def build_volume_features(
    df: pd.DataFrame,
    obv_roc_periods: List[int] | int = 21,
    volume_ratio_periods: List[int] | int = 20,
    mfi_periods: List[int] | int = 14,
    force_index_periods: List[int] | int = 13,
    force_index_norm_window: int = 20,
) -> pd.DataFrame:
    """Build all volume features from the daily DataFrame.

    Args:
        df (pd.DataFrame): Daily DataFrame with 'high', 'low', 'close', 'volume'.
        obv_roc_periods (List[int] | int): Lookback for OBV Rate of Change.
        volume_ratio_periods (List[int] | int): Lookback for Volume Ratio.
        mfi_periods (List[int] | int): Lookback for Money Flow Index.
        force_index_periods (List[int] | int): Smoothing windows for Force Index.
        force_index_norm_window (int): Normalisation window for Force Index.

    Returns:
        pd.DataFrame: Volume features with dynamic column names.
    """
    high, low, close, volume = (
        df["high"].copy(),
        df["low"].copy(),
        df["close"].copy(),
        df["volume"].copy(),
    )

    # OBV Rate of Change
    obv_df = _compute_for_periods(
        period=obv_roc_periods,
        compute_func=lambda p: compute_obv_roc(close, volume, p),
        feat_name="obv_roc",
        index=df.index,
    )

    # Volume Ratio
    vol_ratio_df = _compute_for_periods(
        period=volume_ratio_periods,
        compute_func=lambda p: compute_volume_ratio(volume, p),
        feat_name="volume_ratio",
        index=df.index,
    )

    # Money Flow Index
    mfi_df = _compute_for_periods(
        period=mfi_periods,
        compute_func=lambda p: compute_mfi(high, low, close, volume, p),
        feat_name="mfi",
        index=df.index,
    )

    # Normalised Force Index
    force_df = _compute_for_periods(
        period=force_index_periods,
        compute_func=lambda p: compute_force_index_normalised(
            close, volume, p, force_index_norm_window
        ),
        feat_name="force_index_norm",
        index=df.index,
    )

    return pd.concat([obv_df, vol_ratio_df, mfi_df, force_df], axis=1)
