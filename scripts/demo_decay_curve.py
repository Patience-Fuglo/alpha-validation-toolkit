"""Demo: rolling IC and decay curve of the Day-2 momentum baseline on real
NVDA/MSFT bars -- two different questions about the same signal.

Rolling IC: is this signal's overall skill holding up over calendar time?
Decay curve: given one signal reading, how many days can you wait before
acting on it before it's gone stale?

Run: python scripts/demo_decay_curve.py
"""

from __future__ import annotations

import numpy as np

from quant_toolkit.data import load_ohlcv
from quant_toolkit.metrics import decay_curve, icir, rolling_ic

TICKERS = ["NVDA", "MSFT"]
START, END = "2019-01-01", "2024-12-31"
SIGNAL_HORIZON = 5  # same trailing-return signal used in the Day-2 IC demo
ROLLING_WINDOW = 60
DECAY_HORIZONS = [1, 3, 5, 10, 20, 40]


def main() -> None:
    for ticker in TICKERS:
        print(f"\n{'=' * 60}\n{ticker}\n{'=' * 60}")
        bars = load_ohlcv(ticker, START, END)
        close = bars["close"]
        signal = close.pct_change(SIGNAL_HORIZON)

        forward_return = close.shift(-SIGNAL_HORIZON) / close - 1.0
        roll = rolling_ic(signal, forward_return, window=ROLLING_WINDOW)
        print(f"Rolling {ROLLING_WINDOW}-day IC: mean {np.nanmean(roll):+.4f}  ICIR {icir(roll):+.3f}")
        print(f"  (calendar-time question: is this signal's overall skill holding up?)")

        curve = decay_curve(signal, close, horizons=DECAY_HORIZONS)
        print("\nDecay curve (horizon-in-days -> IC):")
        for h, ic in curve.items():
            print(f"  {h:>3}d: {ic:+.4f}")
        print("  (shelf-life question: how long does one signal reading stay useful?)")


if __name__ == "__main__":
    main()
