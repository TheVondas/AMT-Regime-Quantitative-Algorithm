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

import pandas as pd

# Regime label constants
TRENDING_UP = 0
TRENDING_DOWN = 1
RANGING_NEUTRAL = 2
DISTRIBUTION = 3
ACCUMULATION = 4
TRANSITION = 5

REGIME_NAMES = {
    TRENDING_UP: "Trending Up",
    TRENDING_DOWN: "Trending Down",
    RANGING_NEUTRAL: "Ranging Neutral",
    DISTRIBUTION: "Distribution",
    ACCUMULATION: "Accumulation",
    TRANSITION: "Transition/Breakout",
}


def compute_kama(
    close: pd.Series, n: int = 10, n_s: int = 2, n_l: int = 30
) -> pd.Series:
    """Kaufman's Adaptive Moving Average — adapts speed to market efficiency.

    In trending markets (high efficiency ratio), KAMA tracks price closely
    like a short EMA. In choppy markets (low efficiency ratio), KAMA barely
    moves, avoiding whipsaw signals.

    Args:
        close: Daily closing prices.
        n: Efficiency ratio lookback window (trading days).
        n_s: Fast smoothing period (used when trending).
        n_l: Slow smoothing period (used when ranging).

    Returns:
        KAMA values as a Series, same index as input.
    """
    fast_sc = 2.0 / (n_s + 1)
    slow_sc = 2.0 / (n_l + 1)

    # Efficiency ratio: net movement / total path
    direction = (close - close.shift(n)).abs()
    volatility = close.diff().abs().rolling(window=n).sum()
    # Avoid division by zero on perfectly flat segments
    er = direction / volatility.clip(lower=1e-10)

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
) -> pd.Series:
    """Detect trend direction from KAMA position and slope.

    Trend up: price > KAMA * (1 + dead_zone) AND KAMA slope > min_slope.
    Trend down: price < KAMA * (1 - dead_zone) AND KAMA slope < -min_slope.
    No trend: price near KAMA (within dead zone) or KAMA slope too flat.

    The dead zone prevents whipsaw classification when price oscillates
    around KAMA. The minimum slope prevents flat KAMA from counting as
    directional. Both thresholds are normalised to price level.

    Args:
        close: Daily closing prices.
        kama: KAMA values (from compute_kama).
        slope_window: Number of days to measure KAMA slope over.
        dead_zone_pct: Minimum fractional distance from KAMA to classify
            as trending (default 1% — median close-to-KAMA distance).
        min_slope_pct: Minimum fractional KAMA slope per slope_window
            to classify as directional (default 0.3%).

    Returns:
        Series with values: 1 (up), -1 (down), 0 (no trend).
    """
    # Normalised KAMA slope
    kama_slope_norm = (kama - kama.shift(slope_window)) / kama.clip(lower=1e-10)

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
        high: Daily high prices.
        low: Daily low prices.
        close: Daily closing prices.
        atr_period: ATR lookback period.
        avg_period: SMA period for ATR average.
        vol_threshold: Multiplier for ATR average to trigger high-vol
            classification (default 1.1 = ATR must be 10% above average).

    Returns:
        Series with values: 1 (high vol), 0 (low vol).
    """
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
        trend: Trend direction series (1, -1, 0).
        high_vol: Volatility regime series (1, 0).

    Returns:
        Series with base regime labels.
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
        base_states: Base regime labels from assign_base_states.
        trend: Raw trend direction series (1, -1, 0) from detect_trend.
        prior_window: Number of days to look back for prior context.
        dominance_pct: Fraction of prior window that must show directional
            trend to qualify (default 30%).

    Returns:
        Series with 6 refined regime labels.
    """
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
        labels: Regime label series.
        min_days: Minimum regime duration in trading days.

    Returns:
        Smoothed regime label series.
    """
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


def build_regime_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Build regime labels for every trading day.

    Orchestrates the full labelling pipeline:
    KAMA → trend detection → volatility detection → base states →
    prior context → minimum duration filter.

    Args:
        df: Daily DataFrame with 'high', 'low', 'close' columns.

    Returns:
        DataFrame with columns:
          - regime_id: integer regime label (0-5)
          - regime_label: string regime name
    """
    close = df["close"]
    high = df["high"]
    low = df["low"]

    # Step 1: Compute KAMA
    kama = compute_kama(close, n=10, n_s=2, n_l=30)

    # Step 2: Detect trend direction
    trend = detect_trend(close, kama, slope_window=5)

    # Step 3: Detect volatility regime
    high_vol = detect_volatility(high, low, close, atr_period=14, avg_period=50)

    # Step 4: Assign base states (4 states + transition)
    base_states = assign_base_states(trend, high_vol)

    # Step 5: Refine ranging states with prior context (→ 6 states)
    refined = add_prior_context(base_states, trend, prior_window=20)

    # Step 6: Smooth out short-lived regimes
    labels = apply_min_duration(refined, min_days=5)

    # Build output DataFrame
    result = pd.DataFrame(index=df.index)
    result["regime_id"] = labels
    result["regime_label"] = labels.map(REGIME_NAMES)

    return result
