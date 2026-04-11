# Current Project State

**Last updated:** 2026-04-11

## Current Stage
Stage 1, Week 2 in progress — momentum, trend, volatility, and volume features complete. Stationarity, time-series, and macro features next.

## What's Next
- [x] Write `src/data/download.py` — download SPY, VIX, yields from yfinance ✓
- [x] Write `src/data/clean.py` — align dates, forward-fill, compute log returns ✓
- [x] Save to `data/raw/` and `data/processed/` ✓
- [x] Verification plot: SPY price, VIX, 10Y yield on same time axis ✓
- [x] Week 2 — Momentum features: ROC (4 lookbacks), RSI, CMO, MACD ✓
- [x] Week 2 — Trend features: ADX, +DI/-DI, Price/SMA, SMA crossover ✓
- [x] Week 2 — Volatility features: ATR, rolling std, VIX, VIX change ✓
- [x] Week 2 — Volume features: OBV ROC, volume ratio, MFI, normalised Force Index ✓
- [ ] Week 2 — Stationarity features: rolling ADF
- [ ] Week 2 — Time-series features: time reversal asymmetry
- [ ] Week 2 — Macro features: yield level, yield change, yield curve slope
- [ ] Week 2 — Fractional differencing, 1-day lag, correlation check, save feature matrix

## Active Concerns
- Intraday data source for Stage 2 AMT features: Polygon.io vs Alpaca vs Databento. Decision needed by Week 7
- Whether to include short strategies or go long-only for v1
- Futures (ES) vs ETF (SPY) as primary instrument
- AMT microstructure signals may be attenuated on SPY/ES vs single-name stocks, metals, or crypto — multi-asset robustness testing will validate

## Known Issues
- **`ta` library dependency risk:** Single-maintainer package. Momentum features (RSI, ROC, MACD) currently depend on it. CMO is already pure pandas. Consider rewriting all momentum indicators in pure pandas/numpy to eliminate the dependency. Wrapper pattern in `momentum.py` isolates the risk for now — only internal function bodies would change. Decision: revisit before Stage 2, or sooner if `ta` causes issues. Same concern applies to any `ta` usage in trend/volatility/volume features.

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

### 2026-04-01 — Collaboration setup with Will Edgington
- **Contributor:** Will Edgington joined as co-developer
- **Division:** Dom handles financial/quantitative/market side; Will handles dev tooling, testing infrastructure, setuptools
- **Dev tooling merged (Will's PRs #1-3):** pre-commit hooks (ruff + black), requirements-dev.txt, commitizen, pyproject.toml, dev setup docs in README
- **Git workflow agreed:** Branch-based development with pull requests. Pull before branching, pull again before pushing. Push promptly. Communicate who's working on what to avoid duplication
- **Branch naming:** `dom/<feature-name>` or `will/<feature-name>`
- **Will's next tasks:** setuptools for clean kernel imports, PyTest setup with unit tests for existing code
- **Dom's next tasks:** Week 2 feature engineering (indicators)

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

### 2026-04-01 — Session 4: Collaboration setup and architectural discussion
**What was done:**
- Pulled and merged Will's PRs #1-3 (pre-commit linting, dev tooling, README updates)
- Installed dev dependencies (ruff, black, pre-commit, commitizen) and activated pre-commit hook
- Discussed AMT signal strength across asset classes — SPY/ES may have attenuated microstructure signals vs single stocks, metals, crypto
- Decided on multi-asset robustness testing post-Week 14 (individual stocks, gold/GC, BTC)
- Discussed data cleaning methodology — current pipeline is appropriate; transformations (winsorisation, ATR normalisation, fractional differencing) belong at feature engineering stage, not raw data level
- Established git collaboration workflow with Will: branch-based development, pull requests, communicate work allocation
- Logged collaboration structure and division of responsibilities

**Key takeaways:**
- AMT microstructure features (TPO, delta, volume profile) are expected to be weaker on SPY than on single stocks/metals/crypto due to ETF arbitrage and index diversification. Source microstructure data from ES futures for Stage 2
- Multi-asset testing is the strongest robustness check — if the system only works on one instrument, it's likely overfit
- Raw data cleaning should preserve information; transformations belong in the feature engineering pipeline where they can be tuned per feature
- Git workflow: pull → branch → work → commit → pull again → push → PR → merge. Communicate with Will to avoid working on the same files

### 2026-04-01 — Session 5: Momentum feature engineering
**What was done:**
- Created `src/features/momentum.py` — computes 9 momentum features: ROC (21/63/126/252 day), RSI (14-day), CMO (14-day), MACD line/signal/histogram (12/26/9)
- Created `src/features/verify_momentum.py` — 4-panel verification chart (SPY, RSI, MACD, ROC) with crisis event annotations
- Verified all features against live data: 5,091 usable rows after 252-day warmup, all values in expected ranges
- Installed `ta` library (v0.11.0) for technical indicator calculations
- First full branch-based git workflow executed: created `dom/add-momentum-features`, committed, pushed, created PR, merged on GitHub, pulled back to main
- Discussed docstring standards (Google-style) and code discoverability — agreed that docstrings + consistent naming + IDE search is preferable to a manual function log
- Identified `ta` library as a single-maintainer dependency risk — logged in Known Issues. CMO already implemented in pure pandas; RSI, ROC, MACD can be rewritten if needed

**Key takeaways:**
- Pre-commit hooks caught formatting issues and a too-large PNG (>500KB) — regenerated at lower DPI. These hooks are now part of the workflow
- `PYTHONPATH=.` needed to run feature scripts until Will's setuptools work is merged
- Momentum features visually confirmed: RSI drops below 30 during crises, MACD goes deeply negative during sustained downtrends, ROC_252 captures year-over-year extremes (COVID -47%, recovery +77%)
- Wrapper pattern in momentum.py isolates the `ta` dependency — if the library breaks, only function internals need rewriting, not the rest of the codebase

### 2026-04-02 — Session 6: Trend feature engineering
**What was done:**
- Pulled Will's setuptools update: `__init__.py` files added, `pyproject.toml` configured, `pip install -e .` now enables clean imports without `PYTHONPATH` workaround
- Created `src/features/trend.py` — computes 6 trend features: ADX (14-day), +DI (14-day), -DI (14-day), price/SMA(50) ratio, price/SMA(200) ratio, SMA(50)/SMA(200) crossover ratio
- Created `src/features/verify_trend.py` — 4-panel verification chart (SPY+SMAs, ADX, +DI/-DI, SMA crossover ratio)
- Verified all features against live data: 5,144 usable rows after 200-day warmup, all values in expected ranges
- Second full branch-based git workflow: created `dom/add-trend-features`, committed, pushed, PR created and merged

**Key takeaways:**
- SMA ratios (price/SMA, SMA/SMA) are preferable to raw differences — they normalise across the full price history ($80 in 2005 vs $650 in 2026)
- SMA crossover expressed as continuous ratio rather than binary flag — preserves information about how established the crossover is
- ADX is arguably the single most important feature for regime classification — directly measures trend strength regardless of direction
- +DI/-DI capture the AMT concept of initiative vs responsive activity numerically — when they weave together, neither side controls the auction (ranging)
- Scripts must be run as modules (`python -m src.features.verify_trend`) not directly (`python src/features/verify_trend.py`) due to import path resolution
- Pre-commit Black formatting caught on first commit attempt — re-stage and commit again is the standard recovery

### 2026-04-08 — Session 7: Volatility feature engineering
**What was done:**
- Pulled Will's PyTest setup: `conftest.py`, unit tests for download, clean, momentum, and trend modules, pytest added to `requirements-dev.txt`
- Created `src/features/volatility.py` — computes 7 volatility features: ATR (14-day, 30-day) as percentage of close, rolling std of returns (20-day, 60-day), VIX level, VIX 5-day absolute change, VIX 5-day percentage change
- Created `src/features/verify_volatility.py` — 4-panel verification chart (SPY, ATR%, rolling std, VIX + VIX change)
- Verified all features against live data: 5,284 usable rows after 60-day warmup, all values in expected ranges
- Third full branch-based git workflow: created `dom/add-volatility-features`, committed, pushed, PR created and merged

**Key takeaways:**
- ATR expressed as percentage of close (ATR/close × 100) for comparability across 20-year price history — same normalisation principle as SMA ratios in trend features
- VIX percentage change added alongside absolute change — a +5 point VIX move from 12 (42% spike) is very different from +5 from 40 (12.5%). BorutaSHAP will determine which representation the classifier prefers
- Realised volatility (ATR, rolling std) vs implied volatility (VIX) capture different information — ATR/std measure what happened, VIX measures what the market expects. Both needed
- VIX column exists in both `daily.parquet` and volatility features — must avoid duplication when assembling the final feature matrix
- COVID ATR peaked at ~8% of price daily, VIX spiked 213% in 5 days — confirms extreme event data is captured correctly

### 2026-04-11 — Session 8: Volume feature engineering
**What was done:**
- Created `src/features/volume.py` — computes 4 volume features: OBV rate of change (21-day), volume ratio (vs 20-day avg), MFI (14-day), normalised Force Index (13-day EMA)
- Created `src/features/verify_volume.py` — 4-panel verification chart (SPY, OBV ROC, MFI, volume ratio + Force Index)
- Identified and fixed two stationarity/scaling bugs before committing:
  - Raw OBV replaced with OBV rate of change — raw OBV is cumulative (~0 to ~17B over 20 years), classifier would learn time-dependent thresholds. ROC captures the actual signal (is volume flowing in or out?)
  - Raw Force Index replaced with normalised version — raw Force Index scales with price × volume (~8x larger in 2026 vs 2005). Normalised by (close × 20-day avg volume) to produce a dimensionless, time-comparable measure
- Verified all features against live data: 5,322 usable rows after warmup
- Fourth full branch-based git workflow: created `dom/add-volume-features`, committed, pushed, PR created

**Key takeaways:**
- Non-stationary features are a critical risk — they cause the classifier to learn what year it is, not what the market is doing. OBV and Force Index required explicit normalisation beyond what fractional differencing would handle
- Volume ratio is the cleanest volume feature — self-normalising by construction, no stationarity or scaling issues, and potentially a leading indicator of regime transitions
- MFI adds volume weighting to the RSI concept — divergence between price (near highs) and MFI (declining) is an early distribution signal
- Force Index negative mean (-0.0004) confirms "markets take the stairs up and the elevator down" — selling days are slightly more forceful than buying days
- Same `ta` library dependency risk applies to volume features (OBV, MFI, Force Index use `ta`). Volume ratio is pure pandas

### 2026-04-11 — Session 9: Discord notification workflow and session workflow documentation
**What was done:**
- Created `.github/workflows/notify-discord.yml` — GitHub Action that sends a Discord notification whenever STATE.md is updated on main
- Workflow extracts the latest session log entry and sends it as a formatted embed to the project Discord channel
- Discord webhook URL stored as GitHub repository secret (DISCORD_WEBHOOK_URL) — never exposed in code or logs
- Created `planAndStateLog/SESSION_WORKFLOW.md` — comprehensive session workflow document covering: context review, git workflow, code documentation standards (Google-style docstrings), error/risk analysis requirements, verification steps, commit message conventions, session closure procedures, and a quick reference checklist
- SESSION_WORKFLOW.md codifies all the practices developed across Sessions 1-8 into a single reference document that both contributors should read before each session

**Key takeaways:**
- Webhook URLs must be treated as secrets — if leaked, anyone can post to the channel. Stored as GitHub secret, rotated after accidental exposure in chat
- Workflow triggers only on push to main with changes to `planAndStateLog/STATE.md` — no false notifications from other file changes
- Will now receives automatic session summaries in Discord without checking the repo manually
- SESSION_WORKFLOW.md ensures consistency as the project scales — new sessions follow the same due process regardless of who is working
