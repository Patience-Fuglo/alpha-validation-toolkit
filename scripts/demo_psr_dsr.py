"""Demo: PSR and DSR on real NVDA/MSFT daily returns.

Tests whether each ticker's own realized daily-return Sharpe ratio could
plausibly be luck, both on its own (PSR against a 0 benchmark) and under a
15-trial multiple-testing correction (DSR) -- the same k=15 convention used
elsewhere in this toolkit's validation methodology.

Run: python scripts/demo_psr_dsr.py
"""

from __future__ import annotations

from quant_toolkit.data import load_ohlcv
from quant_toolkit.metrics import deflated_sharpe_ratio, probabilistic_sharpe_ratio, sharpe_ratio

TICKERS = ["NVDA", "MSFT"]
START, END = "2019-01-01", "2024-12-31"
N_TRIALS = 15


def main() -> None:
    for ticker in TICKERS:
        print(f"\n{'=' * 60}\n{ticker}\n{'=' * 60}")
        bars = load_ohlcv(ticker, START, END)
        daily_returns = bars["close"].pct_change().dropna()

        sr = sharpe_ratio(daily_returns)
        psr = probabilistic_sharpe_ratio(daily_returns, benchmark_sr=0.0)
        dsr = deflated_sharpe_ratio(daily_returns, n_trials=N_TRIALS)

        print(f"Daily Sharpe ratio:       {sr:+.4f}")
        print(f"PSR (vs. 0 benchmark):    {psr:.4f}")
        print(f"DSR (k={N_TRIALS} trials):        {dsr:.4f}")


if __name__ == "__main__":
    main()
