"""Demo: purged walk-forward with embargo on real NVDA and MSFT daily bars.

Run: python scripts/demo_purged_walk_forward.py
"""

from __future__ import annotations

from quant_toolkit.data import load_ohlcv
from quant_toolkit.validation import PurgedWalkForward

TICKERS = ["NVDA", "MSFT"]
START, END = "2019-01-01", "2024-12-31"

# 5-day-forward-return label -> a label computed at t depends on prices through t+5.
LABEL_HORIZON = 5
N_SPLITS = 5
EMBARGO_PCT = 0.02


def main() -> None:
    for ticker in TICKERS:
        print(f"\n{'=' * 60}\n{ticker}\n{'=' * 60}")
        bars = load_ohlcv(ticker, START, END)
        print(f"Loaded {len(bars)} daily bars: {bars.index.min().date()} -> {bars.index.max().date()}")

        splitter = PurgedWalkForward(
            n_splits=N_SPLITS, label_horizon=LABEL_HORIZON, embargo_pct=EMBARGO_PCT
        )

        for fold, (train_idx, test_idx) in enumerate(splitter.split(bars.index), start=1):
            print(
                f"  Fold {fold}: "
                f"train [{train_idx.min().date()} -> {train_idx.max().date()}] "
                f"({len(train_idx)} rows)  |  "
                f"test [{test_idx.min().date()} -> {test_idx.max().date()}] "
                f"({len(test_idx)} rows)  |  "
                f"gap before test: {(test_idx.min() - train_idx.max()).days} calendar days"
            )


if __name__ == "__main__":
    main()
