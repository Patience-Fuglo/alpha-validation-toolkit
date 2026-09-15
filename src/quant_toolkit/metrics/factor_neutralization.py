"""Factor neutralization: is a signal's performance genuinely its own, or
just a known market factor wearing a disguise?

A strategy's return stream can be regressed against known, well-documented
factors (market, size, value). What's left after subtracting their effect
-- the regression intercept, alpha -- is the part of performance those
factors can't explain. A strategy with near-zero factor betas, near-zero
R-squared, and a real (nonzero) alpha has genuine, independent edge. A
strategy with a high beta on one factor and high R-squared, with no alpha
left once that's subtracted out, was never really its own thing -- it was
that factor, relabeled.

Ordinary least squares is implemented here via ``numpy.linalg.lstsq`` (a
numerically stable solver for the normal equations) rather than
``statsmodels.OLS`` or ``sklearn.LinearRegression`` -- the regression
mechanics (design matrix, residuals, R-squared, coefficient standard
errors, t-statistics) are built explicitly, not delegated to a stats
library that would hide them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class OLSResult:
    alpha: float
    betas: dict[str, float]
    r_squared: float
    t_stat_alpha: float
    t_stats: dict[str, float]
    n_obs: int

    def is_significant(self, coefficient: str, threshold: float = 1.96) -> bool:
        """Rough two-sided ~5% significance heuristic (|t| > ~1.96), the
        same convention used to read the t-stats above by eye."""
        t = self.t_stat_alpha if coefficient == "alpha" else self.t_stats[coefficient]
        return abs(t) > threshold


def ols_regression(y: pd.Series, X: pd.DataFrame) -> OLSResult:
    """Regress ``y`` on ``X`` with an intercept, aligned by index (NaN rows dropped).

    Raises ValueError if fewer observations remain than parameters being
    estimated (intercept + each column of X).
    """
    aligned = pd.concat([y, X], axis=1, join="inner").dropna()
    y_vals = aligned.iloc[:, 0].to_numpy(dtype=float)
    x_cols = list(X.columns)
    x_vals = aligned[x_cols].to_numpy(dtype=float)

    n = len(y_vals)
    k = x_vals.shape[1] + 1  # + intercept
    if n <= k:
        raise ValueError(f"Need more than {k} observations to fit {k} parameters, got {n}")

    design = np.column_stack([np.ones(n), x_vals])
    beta_hat, _, _, _ = np.linalg.lstsq(design, y_vals, rcond=None)

    fitted = design @ beta_hat
    resid = y_vals - fitted
    ss_res = float((resid**2).sum())
    ss_tot = float(((y_vals - y_vals.mean()) ** 2).sum())
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    dof = n - k
    sigma2 = ss_res / dof
    xtx_inv = np.linalg.pinv(design.T @ design)
    se = np.sqrt(np.clip(sigma2 * np.diag(xtx_inv), 0, None))
    t_stats_all = np.divide(beta_hat, se, out=np.full_like(beta_hat, np.nan), where=se > 0)

    return OLSResult(
        alpha=float(beta_hat[0]),
        betas=dict(zip(x_cols, beta_hat[1:].tolist())),
        r_squared=r_squared,
        t_stat_alpha=float(t_stats_all[0]),
        t_stats=dict(zip(x_cols, t_stats_all[1:].tolist())),
        n_obs=n,
    )


def fama_french_alpha(returns: pd.Series, factors: pd.DataFrame) -> OLSResult:
    """Regress ``returns`` in excess of the risk-free rate on Mkt-RF, SMB, HML.

    ``factors`` must have columns ``mkt_rf, smb, hml, rf`` (e.g. from
    ``quant_toolkit.data.load_fama_french_factors``), all in decimal daily
    return units matching ``returns``.
    """
    required = {"mkt_rf", "smb", "hml", "rf"}
    missing = required - set(factors.columns)
    if missing:
        raise ValueError(f"factors is missing required columns: {sorted(missing)}")

    aligned = pd.concat([returns.rename("returns"), factors], axis=1, join="inner").dropna()
    excess_returns = aligned["returns"] - aligned["rf"]
    return ols_regression(excess_returns, aligned[["mkt_rf", "smb", "hml"]])
