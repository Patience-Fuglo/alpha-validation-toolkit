# alpha-validation-toolkit

A from-scratch statistical validation toolkit for cross-sectional equity
alpha research: purged walk-forward cross-validation with embargo, IC / Rank
IC, PSR / DSR with multiple-testing correction, Fama-French factor
neutralization, and rolling IC / decay curves.

Off-the-shelf tools like `sklearn.model_selection.TimeSeriesSplit` or
`alphalens` make convenient assumptions that don't hold for financial
labels — no forward-looking overlap between train and test, no residual
serial correlation across fold boundaries. Every module here is built
without those shortcuts, so the leakage guards are explicit and auditable
rather than implicit in a library's defaults. Everything is validated
against real daily equity data (not synthetic series), so the failure modes
are the real ones — API rate limits, missing bars, actual serial
correlation — not idealized textbook data.

## Status

| Module | Status |
|---|---|
| Purged walk-forward with embargo | done |
| IC / Rank IC | done |
| PSR / DSR (k=15 multiple-testing correction) | done |
| Fama-French factor neutralization | done |
| Rolling IC / decay curve | not started |

## Purged walk-forward with embargo

`src/quant_toolkit/validation/purged_walk_forward.py`

A plain expanding-window time-series split keeps train strictly before test
chronologically, but that alone under-protects financial labels:

- **Purge** — if a label at time *t* is forward-looking (e.g. "5-day forward
  return"), the last `label_horizon` rows before a test fold have labels that
  peek into the test window. Those rows are dropped from training.
- **Embargo** — return/volatility autocorrelation doesn't vanish the instant a
  test fold ends, so a later fold's expanding training window can still be
  indirectly informed by a just-seen test period. An `embargo_pct`-sized
  buffer after *every* prior test fold is excluded from later training sets.

```python
from quant_toolkit.data import load_ohlcv
from quant_toolkit.validation import PurgedWalkForward

bars = load_ohlcv("NVDA", "2019-01-01", "2024-12-31")
splitter = PurgedWalkForward(n_splits=5, label_horizon=5, embargo_pct=0.02)

for train_idx, test_idx in splitter.split(bars.index):
    ...  # fit on bars.loc[train_idx], evaluate on bars.loc[test_idx]
```

Run the real-data demo:

```bash
pip install -e .
python scripts/demo_purged_walk_forward.py
```

Run the tests:

```bash
pytest tests/
```

## IC / Rank IC

`src/quant_toolkit/metrics/ic.py`

The Information Coefficient is the Pearson correlation between a signal and
the forward return it's meant to predict; Rank IC is the same idea computed
on each series' ranks (Spearman), so a single outlier observation can't
dominate the score the way it can with raw values. Both are implemented
from first principles here — no `pandas.Series.corr`, no
`scipy.stats.pearsonr`/`spearmanr` — and the test suite cross-checks the
results against those reference implementations independently.

`rolling_ic` computes IC (or Rank IC) on each trailing window, and `icir`
summarizes that series as mean / std — a signal with the same average IC as
another but a much higher ICIR is the more consistent, more trustworthy one.

```python
from quant_toolkit.data import load_ohlcv
from quant_toolkit.metrics import icir, information_coefficient, rolling_ic

bars = load_ohlcv("NVDA", "2019-01-01", "2024-12-31")
signal = bars["close"].pct_change(5)
forward_return = bars["close"].shift(-5) / bars["close"] - 1.0

information_coefficient(signal, forward_return)
roll = rolling_ic(signal, forward_return, window=60)
icir(roll)
```

Run the real-data demo:

```bash
python scripts/demo_information_coefficient.py
```

## PSR / DSR

`src/quant_toolkit/metrics/psr_dsr.py`

A Sharpe ratio computed on a finite sample is a point estimate, not a
certainty. The **Probabilistic Sharpe Ratio (PSR)** turns that into a
number: the probability that the *true* Sharpe ratio exceeds a chosen
benchmark, given the sample size and the shape (skewness, kurtosis) of the
return distribution — the same-looking Sharpe is far more believable after
1,000 observations than after 10.

The **Deflated Sharpe Ratio (DSR)** corrects for a different problem: if
this result is the best of *N* things tried, some of "best" is just *N*
chances for luck to show up somewhere. DSR raises PSR's benchmark to the
Sharpe ratio the best of *N* pure-noise trials would be expected to produce
by chance alone (an extreme-value approximation), then asks whether the
actual result still clears that higher, corrected bar.

Both are built from the underlying formulas (Bailey & Lopez de Prado, *The
Sharpe Ratio Efficient Frontier*, 2012; *The Deflated Sharpe Ratio*, 2014)
rather than a packaged library call — including the standard normal CDF
(`math.erf`-based) and its inverse (bisection), not `scipy.stats.norm`.

```python
from quant_toolkit.metrics import deflated_sharpe_ratio, probabilistic_sharpe_ratio

probabilistic_sharpe_ratio(daily_returns, benchmark_sr=0.0)
deflated_sharpe_ratio(daily_returns, n_trials=15)
```

Run the real-data demo:

```bash
python scripts/demo_psr_dsr.py
```

## Factor neutralization

`src/quant_toolkit/metrics/factor_neutralization.py`

Regresses a return stream against known, well-documented factors (market,
size, value) to separate what those factors already explain from what's
genuinely left over. A strategy with near-zero factor betas, near-zero
R-squared, and a real, statistically significant alpha has independent
edge. A strategy with a high beta and high R-squared on a known factor,
with no significant alpha once that's subtracted out, was never really its
own thing — it was that factor, relabeled.

OLS is solved via `numpy.linalg.lstsq` (a numerically stable solver for the
normal equations) rather than `statsmodels.OLS` — the design matrix,
residuals, R-squared, coefficient standard errors and t-statistics are all
computed explicitly. Cross-checked against `sklearn.LinearRegression` in
the tests.

```python
from quant_toolkit.data import load_fama_french_factors, load_ohlcv
from quant_toolkit.metrics import fama_french_alpha

factors = load_fama_french_factors("2019-01-01", "2024-12-31")
returns = load_ohlcv("NVDA", "2019-01-01", "2024-12-31")["close"].pct_change().dropna()

result = fama_french_alpha(returns, factors)
result.alpha, result.betas, result.r_squared, result.is_significant("alpha")
```

Factor data is real, official Fama-French daily 3-factor data from
Kenneth French's public data library — not a synthetic or proxied series.

Run the real-data demo:

```bash
python scripts/demo_factor_neutralization.py
```

## Data

`src/quant_toolkit/data.py` loads real daily OHLCV bars via `yfinance`, with
retry/backoff and disk caching (`data/`, gitignored) to survive Yahoo's
free-tier rate limiting, falling back to Stooq's free daily CSV endpoint if
Yahoo is unreachable.
