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
| IC / Rank IC | not started |
| PSR / DSR (k=15 multiple-testing correction) | not started |
| Fama-French factor neutralization | not started |
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

## Data

`src/quant_toolkit/data.py` loads real daily OHLCV bars via `yfinance`, with
retry/backoff and disk caching (`data/`, gitignored) to survive Yahoo's
free-tier rate limiting, falling back to Stooq's free daily CSV endpoint if
Yahoo is unreachable.
