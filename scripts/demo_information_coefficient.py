"""Demo: IC and Rank IC of a simple momentum baseline on real NVDA/MSFT bars.

This uses a single-name time-series signal (past 5-day return predicting the
next 5-day return) purely to exercise the IC/Rank IC/ICIR functions end to
end on real data. It is a baseline sanity check for the metric
implementations, not a validated cross-sectional alpha signal.

Run: python scripts/demo_information_coefficient.py
"""

from __future__ import annotations

import numpy as np

from quant_toolkit.data import load_ohlcv
from quant_toolkit.metrics import icir, information_coefficient, rank_information_coefficient, rolling_ic

TICKERS = ["NVDA", "MSFT"]
START, END = "2019-01-01", "2024-12-31"
HORIZON = 5  # trailing signal window and forward label window, both 5 days
ROLLING_WINDOW = 60


def main() -> None:
    for ticker in TICKERS:
        print(f"\n{'=' * 60}\n{ticker}\n{'=' * 60}")
        bars = load_ohlcv(ticker, START, END)
        close = bars["close"]

        # signal at t: known-as-of-t trailing HORIZON-day return
        signal = close.pct_change(HORIZON)
        # label at t: HORIZON-day-forward return, realized after t (not known at t)
        forward_return = close.shift(-HORIZON) / close - 1.0

        full_ic = information_coefficient(signal, forward_return)
        full_rank_ic = rank_information_coefficient(signal, forward_return)
        print(f"Full-sample IC:      {full_ic:+.4f}")
        print(f"Full-sample Rank IC: {full_rank_ic:+.4f}")

        roll_ic = rolling_ic(signal, forward_return, window=ROLLING_WINDOW, method="pearson")
        roll_rank_ic = rolling_ic(signal, forward_return, window=ROLLING_WINDOW, method="spearman")
        print(f"\nRolling {ROLLING_WINDOW}-day IC:      mean {np.nanmean(roll_ic):+.4f}  ICIR {icir(roll_ic):+.3f}")
        print(f"Rolling {ROLLING_WINDOW}-day Rank IC: mean {np.nanmean(roll_rank_ic):+.4f}  ICIR {icir(roll_rank_ic):+.3f}")


if __name__ == "__main__":
    main()
