"""Probabilistic Sharpe Ratio (PSR) and Deflated Sharpe Ratio (DSR).

A Sharpe ratio computed on a finite sample is a point estimate, not a
certainty -- the same-looking Sharpe is far more believable after 1,000
observations than after 10. PSR turns that intuition into a number: the
probability that the *true* Sharpe ratio exceeds a chosen benchmark, given
the sample size and the shape (skew, kurtosis) of the return distribution.

DSR answers a related but different question: if this result is the best of
N things you tried, how much of that "best" is just N chances for luck to
show up somewhere? DSR corrects for that by raising PSR's benchmark to the
Sharpe ratio you'd expect the best of N *pure-noise* trials to produce by
chance alone, then asks whether the actual result still clears that
higher, multiple-testing-adjusted bar.

Both are implemented from the underlying formulas (Bailey & Lopez de Prado,
"The Sharpe Ratio Efficient Frontier", 2012; "The Deflated Sharpe Ratio",
2014) rather than a packaged library call. The standard normal CDF is built
from `math.erf`; its inverse is found by bisection rather than a canned
`scipy.stats.norm.ppf`.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

_EULER_MASCHERONI = 0.5772156649015329


def _to_array(returns: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(returns, dtype=float)
    arr = arr[~np.isnan(arr)]
    return arr


def sharpe_ratio(returns: pd.Series | np.ndarray) -> float:
    """Sample Sharpe ratio: mean / sample std (ddof=1), same units as ``returns``.

    Not annualized -- callers pass per-period returns and interpret the
    result in that same period's units.
    """
    x = _to_array(returns)
    if len(x) < 2:
        return float("nan")
    std = x.std(ddof=1)
    if np.isclose(std, 0.0, atol=1e-15):
        return float("nan")
    return float(x.mean() / std)


def _skewness(x: np.ndarray) -> float:
    n = len(x)
    m2 = np.mean((x - x.mean()) ** 2)
    m3 = np.mean((x - x.mean()) ** 3)
    if np.isclose(m2, 0.0, atol=1e-15):
        return 0.0
    return float(m3 / m2**1.5)


def _kurtosis(x: np.ndarray) -> float:
    """Raw (non-excess) kurtosis -- a normal distribution scores 3.0 here."""
    m2 = np.mean((x - x.mean()) ** 2)
    m4 = np.mean((x - x.mean()) ** 4)
    if np.isclose(m2, 0.0, atol=1e-15):
        return 3.0
    return float(m4 / m2**2)


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """Inverse standard normal CDF via bisection (monotonic, so it's exact
    to float precision without needing a rational approximation)."""
    if not (0.0 < p < 1.0):
        raise ValueError("p must be in (0, 1)")
    lo, hi = -40.0, 40.0
    for _ in range(100):
        mid = (lo + hi) / 2.0
        if _norm_cdf(mid) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def probabilistic_sharpe_ratio(returns: pd.Series | np.ndarray, benchmark_sr: float = 0.0) -> float:
    """P(true Sharpe > benchmark_sr), given this sample's size, skew and kurtosis.

    Returns NaN if fewer than 3 observations remain (skew/kurtosis are
    undefined below that), or if the sample has zero variance.
    """
    x = _to_array(returns)
    n = len(x)
    if n < 3:
        return float("nan")

    sr_hat = sharpe_ratio(x)
    if np.isnan(sr_hat):
        return float("nan")

    gamma3 = _skewness(x)
    gamma4 = _kurtosis(x)

    denom_inner = 1.0 - gamma3 * sr_hat + ((gamma4 - 1.0) / 4.0) * sr_hat**2
    if denom_inner <= 0:
        return float("nan")

    z = (sr_hat - benchmark_sr) * math.sqrt(n - 1) / math.sqrt(denom_inner)
    return _norm_cdf(z)


def expected_max_sharpe_under_null(n_trials: int, sr_std: float) -> float:
    """Expected Sharpe ratio of the best of ``n_trials`` pure-noise (SR=0) trials.

    Extreme-value approximation from Bailey & Lopez de Prado (2014). Needs
    at least 2 trials -- "best of one" has no selection effect to correct for.
    """
    if n_trials < 2:
        raise ValueError("n_trials must be >= 2 for a multiple-testing correction")
    if sr_std < 0:
        raise ValueError("sr_std must be >= 0")

    z_a = _norm_ppf(1.0 - 1.0 / n_trials)
    z_b = _norm_ppf(1.0 - 1.0 / (n_trials * math.e))
    return sr_std * ((1.0 - _EULER_MASCHERONI) * z_a + _EULER_MASCHERONI * z_b)


def deflated_sharpe_ratio(
    returns: pd.Series | np.ndarray, n_trials: int, sr_std: float | None = None
) -> float:
    """PSR of ``returns``, benchmarked against the best-of-``n_trials`` noise threshold.

    If ``sr_std`` (the standard deviation of Sharpe ratios across the
    ``n_trials`` actually run) isn't supplied, it's approximated by this
    single trial's own Sharpe-estimator standard error -- the standard
    simplification when only the selected best trial's returns are on hand,
    not the full set of N trials' results.
    """
    x = _to_array(returns)
    n = len(x)
    if n < 3:
        return float("nan")

    if sr_std is None:
        sr_hat = sharpe_ratio(x)
        gamma3 = _skewness(x)
        gamma4 = _kurtosis(x)
        denom_inner = 1.0 - gamma3 * sr_hat + ((gamma4 - 1.0) / 4.0) * sr_hat**2
        if denom_inner <= 0:
            return float("nan")
        sr_std = math.sqrt(denom_inner / (n - 1))

    sr0 = expected_max_sharpe_under_null(n_trials, sr_std)
    return probabilistic_sharpe_ratio(x, benchmark_sr=sr0)
