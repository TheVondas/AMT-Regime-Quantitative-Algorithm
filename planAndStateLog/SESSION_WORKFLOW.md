# Session Workflow — Standard Operating Procedures

This document defines the required workflow for every development session. Read this at the start of each session before beginning any work.

---

## 1. Session Initialisation

Before writing any code, complete these steps in order:

### 1.1 Review Project Context

Read the following files to recover full context and ensure alignment with the project mandate:

- **`planAndStateLog/STATE.md`** — Current progress, what's next, active concerns, known issues. Check the latest session log entry to understand where we left off.
- **`planAndStateLog/DEVELOPMENT_PLAN.md`** — Week-by-week task list with definitions of done. Confirm which week/task we are working on and what the acceptance criteria are.
- **`planAndStateLog/PROJECT_PLAN.md`** — Strategic overview, architecture, and design decisions. Reference if any design questions arise during the session.
- **`planAndStateLog/SESSION_WORKFLOW.md`** — This file. Refresh on due process.

### 1.2 Git Initialisation

Pull the latest changes and create a feature branch:

```bash
git pull origin main                      # Refresh local copy with any contributions
git checkout -b dom/<feature-name>        # Create a new branch for this session's work
```

**Branch naming convention:** `dom/<feature-name>` or `will/<feature-name>`. Use short, descriptive names (e.g., `dom/add-rsi-feature`, `dom/add-volatility-features`).

---

## 2. Development Standards

### 2.1 Code Documentation

Every piece of code must be easily readable by both contributors without explanation.

**Module-level docstring** — required at the top of every `.py` file:

```python
"""
Momentum features for regime classification.

Computes: ROC (4 lookbacks), RSI (14-day), CMO (14-day), MACD (12/26/9).
All features are computed from the daily DataFrame in data/processed/daily.parquet.
"""
```

This tells a reader what the file contains and where the data comes from, without opening any function.

**Function-level docstring** — required on every function (Google-style):

```python
def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index — ratio of recent gains to total moves, scaled 0-100.

    Args:
        close: Daily closing prices.
        period: Lookback window in trading days.

    Returns:
        RSI values as a Series (0-100 scale).
    """
```

Include:
- One-line description of what the function does
- `Args:` section listing each parameter and its meaning
- `Returns:` section describing the output
- Any important context about interpretation or caveats (e.g., stationarity concerns, scale)

**Inline comments** — use sparingly, only where the logic is not self-evident. Well-named functions and variables should make most code readable without comments.

### 2.2 Error and Risk Analysis

Before committing any code, explicitly assess:

**Potential bugs:**
- Are there edge cases that could produce NaN, infinity, or division by zero?
- Is the warmup period correct for rolling/lookback calculations?
- Does the output match expected ranges and distributions?

**Stationarity and scaling:**
- Is the feature stationary (consistent statistical properties over time)?
- If not, does it need normalisation, rate of change, or fractional differencing?
- Will the classifier learn from market behaviour or from the passage of time?

**Dependency risks:**
- Does the code rely on a third-party library that could break, be abandoned, or change its API?
- If so, is the dependency isolated behind a wrapper function so internals can be swapped without affecting the rest of the codebase?
- Document the risk in STATE.md under Known Issues.

**Security considerations:**
- Are any secrets, API keys, or credentials at risk of exposure?
- For any external integrations (webhooks, APIs), is the secret stored securely (e.g., GitHub Secrets, environment variables)?

**Deviation from project mandate:**
- Does this implementation differ from what was specified in DEVELOPMENT_PLAN.md or PROJECT_PLAN.md?
- If so, document the deviation and rationale in the STATE.md Decision Log or Session Log.
- Example: replacing raw OBV with OBV rate of change due to stationarity concerns, or adding VIX percentage change alongside absolute change.

### 2.3 Verification

Every feature or module should have a corresponding verification step:

- **Data features:** Verification plot saved to `notebooks/<feature>_features.png`, overlaid on SPY price with crisis event annotations
- **Numerical check:** Print shape, null counts, summary statistics, and sample rows. Confirm values are in expected ranges.
- **Pre-commit hooks:** ruff (linting) and Black (formatting) run automatically on commit. If they fail, re-stage the auto-fixed files and commit again.

---

## 3. Git Workflow

### 3.1 Example: Adding RSI Feature

```
Session start:
  Read STATE.md, DEVELOPMENT_PLAN.md, PROJECT_PLAN.md, SESSION_WORKFLOW.md

Git initialisation:
  git pull origin main
  git checkout -b dom/add-rsi-feature

Development:
  Write code in src/features/momentum.py
  Write verification plot
  Run and verify outputs
  Assess errors, risks, deviations (Section 2.2)
  Update STATE.md with session log entry

Commit:
  git add src/features/momentum.py planAndStateLog/STATE.md
  git commit -m "feat: add 14-day RSI feature calculation"
  (pre-commit hooks run — if they auto-fix, re-stage and commit again)

Pre-push check:
  git pull origin main

Push:
  git push origin dom/add-rsi-feature

Pull Request:
  Create PR on GitHub with title and description
  Review the diff — confirm everything looks correct
  Merge when satisfied

Post-merge:
  git checkout main
  git pull origin main
```

### 3.2 Commit Message Convention

Follow the Conventional Commits format (enforced by commitizen):

| Prefix | When to use | Example |
|--------|-------------|---------|
| `feat:` | New feature or functionality | `feat: add 14-day RSI feature` |
| `fix:` | Bug fix | `fix: correct ATR normalisation divisor` |
| `docs:` | Documentation changes | `docs: log session 8 in STATE.md` |
| `refactor:` | Code restructuring without behaviour change | `refactor: extract SMA calculation into helper` |
| `test:` | Adding or updating tests | `test: add unit tests for momentum features` |
| `chore:` | Maintenance (dependencies, config) | `chore: add ta to requirements.txt` |
| `experiment:` | Notebook experimentation | `experiment: testing MACD signal thresholds` |

### 3.3 What NOT to Do

- **Never push directly to main** without a PR (except STATE.md-only updates by agreement)
- **Never commit secrets** (API keys, webhook URLs, credentials) to code
- **Never use `git add .` or `git add -A`** — always stage specific files to avoid committing sensitive or unnecessary files
- **Never skip pre-commit hooks** (`--no-verify`) — if they fail, fix the issue
- **Never work on the same files as Will simultaneously** without communicating first

---

## 4. Session Closure

Before ending a session:

### 4.1 Update STATE.md

Add a session log entry with:
- **Session number and date**
- **What was done** — bullet points of concrete deliverables
- **Key takeaways** — insights, decisions, risks discovered, deviations from plan

Update the mutable sections (Current Stage, What's Next, Active Concerns, Known Issues) as needed.

### 4.2 Update DEVELOPMENT_PLAN.md

Tick off any completed tasks in the week-by-week checklist. This keeps the plan in sync with actual progress so either contributor can see at a glance what's done and what remains.

### 4.3 Commit and Push STATE.md and DEVELOPMENT_PLAN.md

```bash
git add planAndStateLog/STATE.md planAndStateLog/DEVELOPMENT_PLAN.md
git commit -m "docs: log session N — brief description"
git push origin main
```

This triggers the Discord notification to Will with the session summary.

### 4.4 Communicate with Will

If your session's work affects anything Will is working on, or if you've identified tasks for him, mention it in Discord directly in addition to the automated notification.

---

## 5. Quick Reference Checklist

```
[ ] Read STATE.md, DEVELOPMENT_PLAN.md, PROJECT_PLAN.md, SESSION_WORKFLOW.md
[ ] git pull origin main
[ ] git checkout -b dom/<feature-name>
[ ] Write code with module-level and function-level docstrings
[ ] Verify: plots, stats, expected ranges
[ ] Assess: bugs, stationarity, dependencies, security, deviations
[ ] git add <specific files>
[ ] git commit -m "prefix: description"
[ ] git pull origin main
[ ] git push origin dom/<feature-name>
[ ] Create PR on GitHub, review, merge
[ ] git checkout main && git pull origin main
[ ] Update STATE.md with session log
[ ] Update DEVELOPMENT_PLAN.md — tick off completed tasks
[ ] Commit and push STATE.md and DEVELOPMENT_PLAN.md
[ ] Communicate with Will if needed
```
