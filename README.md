# AMT-Regime-Quantitative-Algorithm

This project builds a quantitative trading system that decomposes financial markets into discrete structural regimes using Auction Market Theory (AMT) and deploys optimised strategies per regime. The core thesis is that markets cycle through identifiable behavioural states — trending, ranging, and transitional — and that strategies tuned to each state will outperform static approaches.

The architecture extends the work of Pomorski (2024), [*"Construction of Effective Regime-Switching Portfolios Using a Combination of Machine Learning and Traditional Approaches"*](https://discovery.ucl.ac.uk/id/eprint/10192012/) (UCL PhD thesis), which validated a detection-prediction-optimisation pipeline using KAMA+MSR for regime labelling, Random Forest for regime prediction, and Model Predictive Control for portfolio construction. This project diverges from Pomorski by:

- Expanding from 4 regimes (volatility × trend) to 6 regimes incorporating directional context and AMT structural states.
- Grounding regime definitions in Auction Market Theory rather than pure statistical volatility decomposition.
- Integrating market microstructure data (TPO, volume profile, delta, VWAP, open interest) as classifier features alongside traditional momentum and volatility indicators.
- Deploying distinct, optimised strategy templates per regime rather than a single long/short approach.
- Using AMT structural levels (value area high/low, POC) for execution-level risk management, including adaptive stop losses that act as real-time regime change detectors.

The predicted edge is structural alpha from two sources: (1) improved regime transition detection via AMT microstructure features that capture participant behaviour, not just price action, and (2) per-regime strategy optimisation that exploits the distinct statistical properties of each market state.

---

## Development Setup

### 1. Environment & Dependencies
This project requires **Python 3.9+**. Using a virtual environment is highly recommended to keep dependencies isolated.

```Bash
python -m venv venv

# Activate the environment:
source venv/bin/activate       # Linux/macOS (Bash/zsh)
venv\Scripts\activate          # Windows (cmd)
```

#### Install Project and Tooling:
We use an "editable" install so that the `src` module is available globally within your environment.

```Bash
pip install -e .               # Installs project + runtime dependencies
pip install -r requirements-dev.txt  # Installs pytest, ruff, black, etc.
```
*Note: Runtime dependencies (`pandas`, `ta`, `yfinance`, etc.) are handled automatically by `pyproject.toml`. You do not need to install `requirements.txt` manually.*

### 2. Enable Quality Automation
We use `pre-commit` to ensure code remains clean and follows professional standards before it reaches the repository.

```Bash
pre-commit install
pre-commit install --hook-type commit-msg # activate Commitizen checks
```

### 3. Workflow:

#### **Branching & PR**:

1. **Branch:** Create a descriptively named branch: `git checkout -b <your-name>/<description>` (e.g., `will/set-up-setuptools`).
2. **Test:** Run `pytest` locally to catch math or logic errors before pushing.
3. **Push & PR:** git push -u origin <branch-name>. Open a Pull Request on GitHub for review.

#### **Commits**:
We use automated tools to keep the code clean and our history organised. When you run `git commit`, several checks will run:
- **Trailing Whitespace/YAML:** Cleans up invisible formatting errors.
- **Black & Ruff:** Automatically formats your code and catches Python bugs.
- **Large Files:** Prevents accidentally uploading huge files (e.g. venv, models, or market data)

**Common Fix:** if a commit fails due to "Black" or "Trailing Whitespace", it usually means the tool **fixed the file for you**. Simply stage the changes with `git add` and re-run your commit.

**Commit Message Format:**

We use **Commitizen** (Conventional Commits). Your commits message must start with one of these prefixes:

|Prefix|Use Case|Example|
|---|---|---|
|`feat:`|A new feature added outside of a notebook|`feat: add AMT value area calculation`|
|`fix:`|Fixing a bug in the code|`fix: resolve division by zero for RSI`|
|`docs:`|Changes to documentation or added comments|`docs: add dev setup to README.md`|
|`experiment:`|**(Custom)** any experimentation done in notebooks|`experiment: testing vwap strategy in notebooks/vwap_test.ipynb`|
|`refactor:`|Rewriting code without changing behaviour|`refactor: optimise loop in data loader`|
|`chore:`|Maintenance (updating libraries, etc.)|`chore: add pandas to requirements.txt`|
|`test:`|Adding or updating tests|`test: add tests for getRSI method`|

**Handling "Failed" commits:**

If a commit fails and **does not** automatically fix itself, check the terminal output for these common issues:

- **Line Length:** We use an **88-character limit**. If a line is too long (usually due to nested logic or long strings), you'll need to break it up manually.
- **Broad Exceptions (Ruff):** Make sure to give your `except:` statements specific errors (for example `except FileNotFoundError as e:`). **Do not leave them blank.** Ruff will flag these because they can hide critical errors.
- **Unused Imports (Ruff):** Ruff will block commits if you have `import pandas` at the top of a python file but don't use it. Simply delete the unused line.
- **Commit Message format (Commitizen)**: if the error says `commit-msg hook failed`, ensure your message starts with a lowercase prefix and a colon (e.g. `experiment: testing vwap strategy in notebooks/vwap_test.ipynb`).
- **Large file errors on Jupyter notebooks (Ruff):** make sure to clear notebook output before commiting.

*For more details, see the [Conventional Commits Specification](https://www.conventionalcommits.org/en/v1.0.0/) and [Pre-commit Documentation](https://pre-commit.com/).*

---
