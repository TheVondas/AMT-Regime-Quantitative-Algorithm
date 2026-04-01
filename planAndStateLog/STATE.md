# Current Project State

**Last updated:** 2026-03-31

## Current Stage
Stage 1, Week 1 complete — data pipeline built and verified. Ready for Week 2 (feature engineering).

## What's Next
- [x] Write `src/data/download.py` — download SPY, VIX, yields from yfinance ✓
- [x] Write `src/data/clean.py` — align dates, forward-fill, compute log returns ✓
- [x] Save to `data/raw/` and `data/processed/` ✓
- [x] Verification plot: SPY price, VIX, 10Y yield on same time axis ✓
- [ ] Week 2: Pomorski-style feature engineering (~20 features: momentum, trend, volatility, volume)

## Active Concerns
- Intraday data source for Stage 2 AMT features: Polygon.io vs Alpaca vs Databento. Decision needed by Week 7
- Whether to include short strategies or go long-only for v1
- Futures (ES) vs ETF (SPY) as primary instrument
- AMT microstructure signals may be attenuated on SPY/ES vs single-name stocks, metals, or crypto — multi-asset robustness testing will validate

## Known Issues
- None yet

---

# Decision Log (append-only — never delete entries, only add)

### 2026-03-26 — Project inception
- **Classifier:** Random Forest (validated by Pomorski, interpretable via SHAP)
- **Labeller v1:** Rule-based (KAMA/EMA trend + ATR volatility + prior state context). KAMA+MSR is v2 upgrade
- **Starting instrument:** SPY, daily frequency
- **Starting features:** Pomorski-style only. AMT features added in Stage 2
- **Latency target:** End of day. No intraday/HFT
- **Transaction costs:** 0.2% round-trip baseline, sensitivity tested at 0.1-0.8%
- **Alpha definition:** Significant intercept in Fama-French 6-factor regression after costs
- **Stops:** Broker-side orders at AMT structural levels, not algorithm-managed
- **6 regimes:** Trending Up, Trending Down, Ranging Neutral, Ranging After Uptrend (Distribution), Ranging After Downtrend (Accumulation), Transition/Breakout
- **Architecture:** Features → RF classifier → regime probability vector → per-regime strategy → position sizing → execution
- **Transition handling:** Soft probability thresholds + probability rate of change + circuit breaker + AMT structural stops
- **Context management:** Use STATE.md for session continuity. Read before every edit. Decision log is append-only

### 2026-03-31 — Data split decision
- **Training + Validation period:** 2006 — end of 2021 (walk-forward folds within this range)
- **Test Holdout period:** 2022 — 2024 (touched ONCE at Week 13, never before)
- **Split ratio:** ~80/20 (training-heavy to capture multiple market cycles)
- **Walk-forward:** expanding window, 1-year validation folds, 5-day purge gap (PGTS)
- **Rationale:** training covers 2008 GFC, 2020 COVID, multiple bull/bear cycles. Holdout covers 2022 rate shock + 2023-24 recovery. ~3,800 training days (above 1,800 minimum). ~750 holdout days (sufficient for Sortino > 2 strategies)
- **Discipline rule:** holdout data exists in data/raw/ but must not be evaluated against until Week 13

### 2026-03-31 — Extreme event handling strategy
- **Winsorisation:** Cap features at 1st/99th percentile to prevent outlier domination (applied at feature engineering stage)
- **ATR normalisation:** Express price-derived features as multiples of ATR so crisis volatility doesn't create artificial outliers
- **Robust metrics:** Use median returns, Sortino ratio, max drawdown rather than mean/Sharpe which are skew-sensitive
- **Crisis vs non-crisis evaluation:** Report performance separately for crisis periods (GFC, COVID, 2022 rate shock) and non-crisis periods
- **Leave-one-crisis-out:** Train excluding one crisis, test on that crisis, repeat for each major event — tests generalisation to unseen stress regimes
- **Sample weighting:** Regime-equalised weighting so each regime contributes equally to training regardless of temporal duration
- **Stress testing:** Perturb features by ±1σ / ±2σ to verify classifier doesn't flip regimes on noise
- **Key principle:** Never exclude crisis data — these periods contain the highest-value regime transitions. Treat them as signal, not noise

### 2026-04-01 — Multi-asset robustness testing
- **Primary instrument:** SPY (build and validate the full pipeline here first)
- **Robustness test instruments:** Individual stocks, metals (e.g. gold/GC), cryptocurrencies (e.g. BTC) — asset classes where AMT microstructure signals are expected to be strongest due to higher volatility and more order-driven price discovery
- **Purpose:** If the system only works on SPY, it may be overfit to one instrument's statistical properties. Generalisation across asset classes with different microstructure characteristics is stronger evidence of genuine signal
- **AMT signal strength by asset class (expected):** Single stocks / crypto / metals > ES futures > SPY ETF. ETF arbitrage mechanics and index diversification attenuate raw auction signals on SPY
- **Practical implication for Stage 2:** Source AMT microstructure data (TPO, delta, volume profile) from ES futures rather than SPY, where genuine price discovery occurs. SPY follows ES, not the other way around (Hasbrouck 2003)
- **Timing:** Multi-asset testing after the SPY pipeline is validated end-to-end (post-Week 14). Not a blocker for Stages 1-4

---

# Session Log (append-only — one entry per working session)

### 2026-03-26 — Session 1: Planning
**What was done:**
- Analysed Pomorski (2024) UCL thesis in full — extracted methodology, features, hyperparameters, results
- Designed 6-state regime model grounded in AMT
- Discussed feasibility, latency, data sourcing, AMT feature integration, alpha measurement, overfitting risks
- Wrote PROJECT_PLAN.md (strategic overview)
- Wrote DEVELOPMENT_PLAN.md (week-by-week execution with checklists)
- Set up memory files for cross-session persistence

**Key takeaways:**
- Alpha comes from per-regime strategy optimisation, not regime detection itself (detection is infrastructure)
- AMT microstructure features (TPO, VP, delta, VWAP) capture participant behaviour that price-derived features miss — especially at regime transitions
- Adaptive stops at AMT structural levels (VAH/VAL) act as execution-level regime change detectors independent of the classifier
- Pomorski's RF achieved MCC ~0.43, Sortino 4.7-25.0 — sets the benchmark to match or beat

### 2026-03-31 — Session 2: Environment setup and data split design
**What was done:**
- Fixed directory structure (folders were incorrectly nested, now siblings as intended)
- Created `.gitignore` (excludes venv/, data files, pycache, IDE files)
- Created virtual environment (`venv/`) with Python 3.14.3
- Installed Week 1 dependencies: pandas, numpy, yfinance, matplotlib, pyarrow
- Saved `requirements.txt`
- Tested yfinance yield tickers: ^TNX (10Y), ^IRX (13-week T-bill), ^FVX (5Y) all confirmed working
- Designed and documented data split strategy (training 2006-2021, holdout 2022-2024)
- Added Appendix A1 (directory layout), A2 (virtual environments), A3 (data splits and overfitting), A4 (handling extreme macro-shock events) to DEVELOPMENT_PLAN.md
- Designed and documented extreme event handling strategy (7 techniques across feature, training, and evaluation stages)

**Key takeaways:**
- No academic formula determines the optimal train/test split — it's constrained judgement based on: sufficient samples per class, regime coverage in both sets, temporal ordering, and purge gaps
- The holdout discipline rule is critical: 2022-2024 data must not be touched until Week 13
- Walk-forward with PGTS purging is the standard approach for financial time series (used by Pomorski, supported by de Prado 2018)
- All data can be sourced from yfinance — no FRED API key needed for v1
- Crisis periods (GFC, COVID, 2022) must never be excluded — they contain the highest-value regime transitions. Winsorisation + ATR normalisation handle the feature-level impact; leave-one-crisis-out tests generalisation

### 2026-03-31 — Session 3: Data pipeline implementation
**What was done:**
- Wrote `src/data/download.py` — downloads SPY, ^VIX, ^TNX, ^IRX, ^FVX from yfinance, saves as Parquet to data/raw/
- Wrote `src/data/clean.py` — aligns all series to SPY's trading day index, forward-fills yield gaps (limit 2 days), computes log returns, saves to data/processed/daily.parquet
- Wrote `src/data/verify_plot.py` — 3-panel verification chart (SPY, VIX, 10Y yield) with crisis event annotations and train/holdout split marker
- All scripts run successfully: 5,343 clean daily rows from 2005-01-04 to 2026-03-31

**Key takeaways:**
- US 2Y yield not available on yfinance — substituted US 5Y (^FVX). Can add 2Y from FRED later if yield curve inversion signal needed (10Y-3M spread from ^TNX/^IRX is the Fed's preferred recession indicator anyway)
- Only 1 row dropped in cleaning (first day, no prior close for log return) — data quality is excellent
- auto_adjust=True in yfinance gives split/dividend-adjusted prices, critical for 20+ year SPY series
- VIX hit 82.7 during COVID, US 3M T-bill briefly went negative (-0.105) in 2020 — both real phenomena, not data errors
