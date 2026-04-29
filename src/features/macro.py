"""
Macro features for regime classification.

Computes: 10Y yield level (passthrough), 10Y yield 20-day change, 10Y yield
percentage change, yield curve slope (10Y-3M and 10Y-5Y).

These are the only features that give the classifier direct visibility into
the macroeconomic environment rather than just price/volume dynamics:

  - Yield level: reflects the rate environment — low rates favour risk assets
    (trending up), high/rising rates pressure valuations (distribution/transition).
  - Yield change: captures the speed of rate moves. Rapid rises (2022 rate shock)
    break regimes; gradual moves are absorbed.
  - Yield curve slope (10Y-3M): the Fed's preferred recession indicator. Positive
    = normal economy (expansion, trending up likely). Inverted (negative) =
    recession expectations (pre-crisis, distribution). Inverted before both
    the GFC and 2020/2022.
  - Yield curve slope (10Y-5Y): secondary slope measure capturing term premium
    shifts in the mid-to-long end, less sensitive to Fed policy than 10Y-3M.

Note: the DEVELOPMENT_PLAN originally specified 10Y-2Y slope, but the US 2Y
yield is not available on yfinance. 10Y-3M (from ^TNX and ^IRX) is substituted
as the primary slope — this is the spread the Fed itself uses for recession
monitoring. 10Y-5Y (from ^TNX and ^FVX) is added as a secondary measure.
This deviation was identified in Session 3 and documented here.

All features are computed from the daily DataFrame in data/processed/daily.parquet.
"""

import pandas as pd


def compute_yield_change(yield_series: pd.Series, period: int = 20) -> pd.Series:
    """Absolute change in yield level over a lookback period.

    Captures the speed and direction of rate moves. A +50bps move in 20 days
    is a very different macro signal than a stable yield at the same level.

    Args:
        yield_series (pd.Series): Daily yield closing values (e.g., us10y).
        period (int): Lookback period in trading days (default 20).

    Returns:
        pd.Series: Absolute change in yield over the period (in percentage points).
    """
    assert period > 0, f"period ({period}) must be a positive integer."
    return yield_series.diff(period)


def compute_yield_pct_change(yield_series: pd.Series, period: int = 20) -> pd.Series:
    """Percentage change in yield level over a lookback period.

    Percentage change normalises for the yield level — a +50bps move from
    1.5% (33% increase) is a much larger shock than +50bps from 4.5% (11%).
    This matters because regime impact scales with the relative magnitude
    of the move, not just the absolute basis points.

    Args:
        yield_series (pd.Series): Daily yield closing values (e.g., us10y).
        period (int): Lookback period in trading days (default 20).

    Returns:
        pd.Series: Percentage change in yield over the period (decimal, not multiplied
        by 100).
    """
    assert period > 0, f"period ({period}) must be a positive integer."
    return yield_series.pct_change(period)


def compute_yield_curve_slope(
    long_yield: pd.Series, short_yield: pd.Series
) -> pd.Series:
    """Yield curve slope — difference between long and short term yields.

    Positive slope = normal (economy expanding, long-term rates higher than
    short-term). Negative slope = inverted (recession signal, short-term
    rates exceed long-term due to tight monetary policy and recession
    expectations).

    Args:
        long_yield (pd.Series): Long-term yield series (e.g., 10Y).
        short_yield (pd.Series): Short-term yield series (e.g., 3M or 5Y).

    Returns:
        pd.Series: Yield curve slope in percentage points (long - short).
    """
    return long_yield - short_yield


def build_macro_features(
    df: pd.DataFrame, us10y_change_period: int = 20
) -> pd.DataFrame:
    """Build all macro features from the daily DataFrame.

    Args:
        df (pd.DataFrame): Daily DataFrame with 'us10y', 'us5y', and 'us3m' columns.
        us10y_change_period (int): Lookback for 10Y change features (default 20).

    Returns:
        pd.DataFrame: Macro features with dynamic column names:
            - us10y
            - us10y_change_{us10y_change_period}d
            - us10y_pct_change_{us10y_change_period}d
            - yield_curve_10y3m
            - yield_curve_10y5y
    """
    us10y = df["us10y"]
    us5y = df["us5y"]
    us3m = df["us3m"]

    features = pd.DataFrame(index=df.index)

    # 10Y yield level (passthrough)
    features["us10y"] = us10y

    # 10Y yield 20-day change (absolute and percentage)
    features[f"us10y_change_{us10y_change_period}d"] = compute_yield_change(
        yield_series=us10y, period=us10y_change_period
    )
    features[f"us10y_pct_change_{us10y_change_period}d"] = compute_yield_pct_change(
        yield_series=us10y, period=us10y_change_period
    )

    # Yield curve slopes
    features["yield_curve_10y3m"] = compute_yield_curve_slope(us10y, us3m)
    features["yield_curve_10y5y"] = compute_yield_curve_slope(us10y, us5y)

    return features
