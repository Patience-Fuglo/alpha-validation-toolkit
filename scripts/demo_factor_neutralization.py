"""Demo: Fama-French 3-factor neutralization of real NVDA/MSFT daily returns.

Regresses each ticker's excess daily return against real Mkt-RF, SMB, HML
factor data (Kenneth French's public data library) and reports beta, R-squared
and alpha -- how much of the stock's return the known market/size/value
factors explain, and what's left over once they're subtracted out.

Run: python scripts/demo_factor_neutralization.py
"""

from __future__ import annotations

from quant_toolkit.data import load_fama_french_factors, load_ohlcv
from quant_toolkit.metrics import fama_french_alpha

TICKERS = ["NVDA", "MSFT"]
START, END = "2019-01-01", "2024-12-31"


def main() -> None:
    factors = load_fama_french_factors(START, END)

    for ticker in TICKERS:
        print(f"\n{'=' * 60}\n{ticker}\n{'=' * 60}")
        bars = load_ohlcv(ticker, START, END)
        daily_returns = bars["close"].pct_change().dropna()

        result = fama_french_alpha(daily_returns, factors)

        print(f"n_obs: {result.n_obs}")
        print(f"alpha (daily): {result.alpha:+.5f}  (t={result.t_stat_alpha:+.2f}, "
              f"significant={result.is_significant('alpha')})")
        for factor, beta in result.betas.items():
            t = result.t_stats[factor]
            print(f"beta[{factor}]: {beta:+.3f}  (t={t:+.2f}, significant={result.is_significant(factor)})")
        print(f"R-squared: {result.r_squared:.4f}")


if __name__ == "__main__":
    main()
