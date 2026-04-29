"""
Rule-based regime labeller (v1) for ground truth generation.

Assigns one of 6 regime labels to every trading day using four layers:

  1. Trend direction — KAMA (Kaufman's Adaptive Moving Average) position
     and slope with dead zone. KAMA adapts its smoothing speed to market
     efficiency: fast in trends, nearly frozen in choppy markets. A 1%
     dead zone around KAMA and a 0.3% minimum slope threshold prevent
     whipsaw signals and flat-KAMA noise.

  2. Volatility level — ATR(14) vs 1.1× its 50-day SMA. The 10%
     threshold prevents marginal above-average vol from inflating the
     Transition category.

  3. Prior-state context — distinguishes three ranging sub-types
     (distribution, accumulation, neutral) based on the dominant raw
     trend signal in the prior 20 days (30% dominance threshold).

  4. Minimum duration filter — regimes lasting < 5 days are absorbed
     into the surrounding regime to reduce label noise.

The 6 regimes:
  0 = Trending Up        — price > KAMA + dead zone, KAMA rising
  1 = Trending Down      — price < KAMA - dead zone, KAMA falling
  2 = Ranging Neutral    — price near KAMA, no prior directional bias
  3 = Distribution       — ranging after uptrend (Wyckoff distribution)
  4 = Accumulation       — ranging after downtrend (Wyckoff accumulation)
  5 = Transition/Breakout — no trend + high volatility

KAMA parameters from Pomorski (2024): n=10, n_s=2, n_l=30.
KAMA+MSR is the v2 upgrade path (see Decision Log).

All labels are computed from the daily DataFrame in data/processed/daily.parquet.
"""

from typing import Dict

import pandas as pd

# Regime label constants
TRENDING_UP = 0
TRENDING_DOWN = 1
RANGING_NEUTRAL = 2
DISTRIBUTION = 3
ACCUMULATION = 4
TRANSITION = 5

REGIME_NAMES: Dict[int, str] = {
    TRENDING_UP: "Trending Up",
    TRENDING_DOWN: "Trending Down",
    RANGING_NEUTRAL: "Ranging Neutral",
    DISTRIBUTION: "Distribution",
    ACCUMULATION: "Accumulation",
    TRANSITION: "Transition/Breakout",
}


def compute_kama(
    close: pd.Series,
    n: int = 10,
    n_s: int = 2,
    n_l: int = 30,
    eps: float = 1e-10,
) -> pd.Series:
    """Kaufman's Adaptive Moving Average — adapts speed to market efficiency.

    In trending markets (high efficiency ratio), KAMA tracks price closely
    like a short EMA. In choppy markets (low efficiency ratio), KAMA barely
    moves, avoiding whipsaw signals.

    Args:
        close (pd.Series): Daily closing prices.
        n (int): Efficiency ratio lookback window (trading days).
        n_s (int): Fast smoothing period (used when trending).
        n_l (int): Slow smoothing period (used when ranging).
        eps (float): Arbitrarily small number where eps > 0.
            Used to avoid division by zero.

    Returns:
        pd.Series: KAMA values, same index as input.
    """
    assert n > 0, f"n ({n}) must be a positive integer."
    assert (
        0 < n_s < n_l
    ), f"n_s ({n_s}) must be a positive integer less than n_l ({n_l})."
    assert (
        0 < eps < 1e-2
    ), f"eps ({eps}) must be an arbitrarily small number > 0. In range: (0, 1e-2)"
    fast_sc = 2.0 / (n_s + 1)
    slow_sc = 2.0 / (n_l + 1)

    # Efficiency ratio: net movement / total path
    direction = (close - close.shift(n)).abs()
    volatility = close.diff().abs().rolling(window=n).sum()
    # Avoid division by zero on perfectly flat segments
    er = direction / volatility.clip(lower=eps)

    # Smoothing constant: maps ER onto [slow_sc, fast_sc], then squares
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2

    # Build KAMA iteratively (depends on previous value)
    kama = pd.Series(index=close.index, dtype=float)
    # Initialise KAMA at the first available close after warmup
    first_valid = n
    kama.iloc[first_valid] = close.iloc[first_valid]

    for i in range(first_valid + 1, len(close)):
        kama.iloc[i] = kama.iloc[i - 1] + sc.iloc[i] * (
            close.iloc[i] - kama.iloc[i - 1]
        )

    return kama


def detect_trend(
    close: pd.Series,
    kama: pd.Series,
    slope_window: int = 5,
    dead_zone_pct: float = 0.01,
    min_slope_pct: float = 0.003,
    eps: float = 1e-10,
) -> pd.Series:
    """Detect trend direction from KAMA position and slope.

    Trend up: price > KAMA * (1 + dead_zone) AND KAMA slope > min_slope.
    Trend down: price < KAMA * (1 - dead_zone) AND KAMA slope < -min_slope.
    No trend: price near KAMA (within dead zone) or KAMA slope too flat.

    The dead zone prevents whipsaw classification when price oscillates
    around KAMA. The minimum slope prevents flat KAMA from counting as
    directional. Both thresholds are normalised to price level.

    Args:
        close (pd.Series): Daily closing prices.
        kama (pd.Series): KAMA values (from compute_kama).
        slope_window (int): Number of days to measure KAMA slope over (default 5).
        dead_zone_pct (float): Minimum fractional distance from KAMA to classify
            as trending (default 1% — median close-to-KAMA distance).
        min_slope_pct (float): Minimum fractional KAMA slope per slope_window
            to classify as directional (default 0.3%).
        eps (float): Arbitrarily small number where eps > 0.
            Used to avoid division by zero.

    Returns:
        pd.Series: Series with values: 1 (up), -1 (down), 0 (no trend).
    """
    assert (
        slope_window > 0
    ), f"slope_window ({slope_window}) must be a positive integer."
    assert (
        0 < eps < 1e-2
    ), f"eps ({eps}) must be an arbitrarily small number > 0. In range: (0, 1e-2)."
    assert (
        0 < min_slope_pct < 1
    ), f"min_slope_pct ({min_slope_pct}) must be in range: (0, 1)."
    assert (
        0 < dead_zone_pct < 1
    ), f"dead_zone_pct ({dead_zone_pct}) must be in range: (0, 1)."
    # Normalised KAMA slope
    kama_slope_norm = (kama - kama.shift(slope_window)) / kama.clip(lower=eps)

    # Price position relative to KAMA with dead zone
    above_band = close > kama * (1 + dead_zone_pct)
    below_band = close < kama * (1 - dead_zone_pct)

    # Slope must be clearly directional
    slope_up = kama_slope_norm > min_slope_pct
    slope_down = kama_slope_norm < -min_slope_pct

    trend = pd.Series(0, index=close.index, dtype=int)
    trend[above_band & slope_up] = 1
    trend[below_band & slope_down] = -1

    return trend


def detect_volatility(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    atr_period: int = 14,
    avg_period: int = 50,
    vol_threshold: float = 1.1,
) -> pd.Series:
    """Detect high/low volatility regime from ATR vs its moving average.

    High volatility: ATR(14) > vol_threshold × SMA(ATR(14), 50).
    Low volatility: ATR(14) ≤ vol_threshold × SMA(ATR(14), 50).

    The threshold (default 1.1) prevents marginal above-average vol
    from being classified as high-vol. Only meaningfully elevated
    volatility triggers the high-vol flag.

    Args:
        high (pd.Series): Daily high prices.
        low (pd.Series): Daily low prices.
        close (pd.Series): Daily closing prices.
        atr_period (int): ATR lookback period.
        avg_period (int): SMA period for ATR average.
        vol_threshold (float): Multiplier for ATR average to trigger high-vol
            classification (default 1.1 = ATR must be 10% above average).

    Returns:
        pd.Series: with values: 1 (high vol), 0 (low vol).
    """
    assert atr_period > 0, f"atr_period ({atr_period}) must be a positive integer."
    assert avg_period > 0, f"atr_period ({avg_period}) must be a positive integer."
    assert (
        vol_threshold >= 1.0
    ), f"vol_threshold ({vol_threshold}) must be at least 1.0."
    # True Range
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.rolling(window=atr_period).mean()
    atr_avg = atr.rolling(window=avg_period).mean()

    high_vol = (atr > vol_threshold * atr_avg).astype(int)
    return high_vol


def assign_base_states(trend: pd.Series, high_vol: pd.Series) -> pd.Series:
    """Combine trend direction and volatility into base states.

    Trend up (any vol)      → Trending Up (0)
    Trend down (any vol)    → Trending Down (1)
    No trend + high vol     → Transition/Breakout (5)
    No trend + low vol      → Ranging Neutral (2, refined later)

    Rationale: directional trend overrides volatility level. Slow grinds
    lower (down + low vol) and momentum rallies (up + high vol) are genuine
    trends. Volatility only matters when there is no clear trend — high vol
    with no trend indicates a regime transition, while low vol with no trend
    indicates a ranging/balance market.

    Args:
        trend (pd.Series): Trend direction series (1, -1, 0).
        high_vol (pd.Series): Volatility regime series (1, 0).

    Returns:
        pd.Series: with base regime labels.
    """
    state = pd.Series(RANGING_NEUTRAL, index=trend.index, dtype=int)

    # Directional trend overrides volatility
    state[trend == 1] = TRENDING_UP
    state[trend == -1] = TRENDING_DOWN

    # No trend: volatility distinguishes transition from ranging
    no_trend = trend == 0
    state[no_trend & (high_vol == 1)] = TRANSITION
    # no_trend & low_vol stays RANGING_NEUTRAL (refined in next step)

    return state


def add_prior_context(
    base_states: pd.Series,
    trend: pd.Series,
    prior_window: int = 20,
    dominance_pct: float = 0.3,
) -> pd.Series:
    """Refine ranging states using prior-state context.

    Ranging after sustained uptrend → Distribution (3)
    Ranging after sustained downtrend → Accumulation (4)
    Ranging after mixed/no trend → Ranging Neutral (2)

    Uses the raw trend signal (1/-1/0) rather than base states to detect
    prior directional bias. This captures bearish/bullish intent even on
    days that didn't meet the strict trending-state criteria (e.g. a day
    with trend=-1 but high vol that became Transition in base states).

    Args:
        base_states (pd.Series): Base regime labels from assign_base_states.
        trend (pd.Series): Raw trend direction series (1, -1, 0) from detect_trend.
        prior_window (int): Number of days to look back for prior context.
        dominance_pct (float): Fraction of prior window that must show directional
            trend to qualify (default 30%).

    Returns:
        pd.Series: 6 refined regime labels.
    """
    assert prior_window > 0, f"prior_window ({prior_window}) must be positive."
    assert (
        0 < dominance_pct < 1
    ), f"dominance_pct ({dominance_pct}) must be in range: (0, 1)."
    refined = base_states.copy()
    threshold = prior_window * dominance_pct

    for i in range(prior_window, len(base_states)):
        if base_states.iloc[i] not in (RANGING_NEUTRAL, TRANSITION):
            continue

        # Use raw trend signal for prior context
        prior_trend = trend.iloc[i - prior_window : i]
        up_count = (prior_trend == 1).sum()
        down_count = (prior_trend == -1).sum()

        if base_states.iloc[i] == RANGING_NEUTRAL:
            if up_count > threshold and up_count > down_count:
                refined.iloc[i] = DISTRIBUTION
            elif down_count > threshold and down_count > up_count:
                refined.iloc[i] = ACCUMULATION
            # else stays RANGING_NEUTRAL
        # Transition stays as-is (no reclassification)

    return refined


def apply_min_duration(labels: pd.Series, min_days: int = 5) -> pd.Series:
    """Absorb regimes shorter than min_days into surrounding regime.

    Short-lived regimes are label noise — a 2-day "Trending Down" inside
    a multi-month "Trending Up" is not a genuine regime change. This
    smoothing pass forward-fills regimes shorter than the minimum.

    Args:
        labels (pd.Series): Regime label series.
        min_days (int): Minimum regime duration in trading days.

    Returns:
        pd.Series: Smoothed regime label series.
    """
    assert min_days >= 1, f"min_days ({min_days}) must be at least 1."
    smoothed = labels.copy()

    # Identify regime runs
    changes = smoothed != smoothed.shift(1)
    run_starts = changes[changes].index.tolist()

    if len(run_starts) < 2:
        return smoothed

    # For each run, check duration
    for i in range(len(run_starts)):
        start = run_starts[i]
        end = run_starts[i + 1] if i + 1 < len(run_starts) else smoothed.index[-1]

        start_pos = smoothed.index.get_loc(start)
        end_pos = smoothed.index.get_loc(end)
        duration = end_pos - start_pos

        if duration < min_days:
            # Absorb into previous regime
            if start_pos > 0:
                prev_label = smoothed.iloc[start_pos - 1]
                smoothed.iloc[start_pos:end_pos] = prev_label

    return smoothed


def build_regime_labels(
    df: pd.DataFrame,
    kama_n: int = 10,
    kama_n_s: int = 2,
    kama_n_l: int = 30,
    slope_window: int = 5,
    dead_zone_pct: float = 0.01,
    min_slope_pct: float = 0.003,
    atr_period: int = 14,
    vol_avg_period: int = 50,
    vol_threshold: float = 1.1,
    context_window: int = 20,
    min_duration: int = 5,
) -> pd.DataFrame:
    """Build regime labels for every trading day.

    Orchestrates the full labelling pipeline:
    KAMA → trend detection → volatility detection → base states →
    prior context → minimum duration filter.

    Args:
        df (pd.DataFrame): Daily DataFrame with 'high', 'low', 'close'.
        kama_n (int): Efficiency ratio lookback window (default 10).
        kama_n_s (int): KAMA fast smoothing period (default 2).
        kama_n_l (int): KAMA slow smoothing period (default 30).
        slope_window (int): Window for measuring KAMA trajectory (default 5).
        dead_zone_pct (float): Fractional distance from KAMA
            required to be 'trending' (default 0.01).
        min_slope_pct (float): Fractional KAMA slope
            required for directionality (default 3e-3).
        atr_period (int): Lookback for Average True Range (default 14).
        vol_avg_period (int): Lookback for the baseline
            average volatility (default 50).
        vol_threshold (float): Multiplier for ATR/SMA(ATR)
            to trigger 'High Vol' (default 1.1).
        context_window (int): Lookback for identifying
            Accumulation/Distribution (default 20).
        min_duration (int): Minimum days a regime must
            persist to avoid being absorbed (default 5).

    Returns:
        pd.DataFrame: With 'regime_id' (int) and 'regime_label' (str).
    """
    close, high, low = df["close"].copy(), df["high"].copy(), df["low"].copy()

    # Step 1: Compute KAMA
    kama = compute_kama(close, n=kama_n, n_s=kama_n_s, n_l=kama_n_l)

    # Step 2: Detect trend direction
    trend = detect_trend(
        close,
        kama,
        slope_window=slope_window,
        dead_zone_pct=dead_zone_pct,
        min_slope_pct=min_slope_pct,
    )

    # Step 3: Detect volatility regime
    high_vol = detect_volatility(
        high,
        low,
        close,
        atr_period=atr_period,
        avg_period=vol_avg_period,
        vol_threshold=vol_threshold,
    )

    # Step 4: Assign base states (4 states + transition)
    base_states = assign_base_states(trend, high_vol)

    # Step 5: Refine ranging states with prior context (→ 6 states)
    refined = add_prior_context(base_states, trend, prior_window=context_window)

    # Step 6: Smooth out short-lived regimes
    labels = apply_min_duration(refined, min_days=min_duration)

    # Build output DataFrame
    result = pd.DataFrame(index=df.index)
    result["regime_id"] = labels
    result["regime_label"] = labels.map(REGIME_NAMES)

    return result
