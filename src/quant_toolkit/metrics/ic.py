"""Information Coefficient (IC) and Rank IC, built from scratch.

IC measures how closely a signal's cross-sectional (or, here, time-series)
ordering matches the ordering of what actually happened next: the Pearson
correlation between a signal and the forward return it's meant to predict.
Rank IC is the same idea computed on ranks instead of raw values (Spearman),
so one outlier observation can't dominate the score the way it can with
Pearson on raw magnitudes.

Both are implemented here from first principles (no `pandas.Series.corr`,
no `scipy.stats.pearsonr`/`spearmanr`) -- the tests cross-check the results
against those reference implementations independently.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def _paired_arrays(signal: pd.Series, forward_return: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Align two series by index and drop any row where either side is NaN."""
    aligned = pd.concat([signal, forward_return], axis=1, join="inner").dropna()
    return aligned.iloc[:, 0].to_numpy(dtype=float), aligned.iloc[:, 1].to_numpy(dtype=float)


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    x_dev = x - x.mean()
    y_dev = y - y.mean()
    denom = np.sqrt((x_dev**2).sum() * (y_dev**2).sum())
    if denom == 0:
        return float("nan")
    return float((x_dev * y_dev).sum() / denom)


def _average_rank(x: np.ndarray) -> np.ndarray:
    """1-indexed ranks with ties broken by averaging the tied positions."""
    order = np.argsort(x, kind="mergesort")
    sorted_x = x[order]
    ranks = np.empty(len(x), dtype=float)
    i = 0
    n = len(x)
    while i < n:
        j = i
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2.0  # average of ranks i+1..j
        i = j
    return ranks


def information_coefficient(signal: pd.Series, forward_return: pd.Series) -> float:
    """Pearson correlation between ``signal`` and ``forward_return``.

    Rows are aligned by index; any row where either series is NaN is dropped.
    Returns NaN if fewer than 2 paired observations remain, or if either
    series is constant (zero variance -> undefined correlation).
    """
    x, y = _paired_arrays(signal, forward_return)
    return _pearson(x, y)


def rank_information_coefficient(signal: pd.Series, forward_return: pd.Series) -> float:
    """Spearman rank correlation between ``signal`` and ``forward_return``.

    Same alignment/NaN handling as ``information_coefficient``, but computed
    on each series' average ranks -- robust to outliers in either series.
    """
    x, y = _paired_arrays(signal, forward_return)
    if len(x) < 2:
        return float("nan")
    return _pearson(_average_rank(x), _average_rank(y))


def rolling_ic(
    signal: pd.Series,
    forward_return: pd.Series,
    window: int,
    method: str = "pearson",
    min_periods: int | None = None,
) -> pd.Series:
    """IC (or Rank IC) computed on each trailing ``window``-length slice.

    Returns a Series indexed like the aligned input, NaN wherever fewer than
    ``min_periods`` (default: ``window``) paired observations are available.
    """
    if method not in ("pearson", "spearman"):
        raise ValueError("method must be 'pearson' or 'spearman'")
    if window < 2:
        raise ValueError("window must be >= 2")
    min_periods = window if min_periods is None else min_periods

    aligned = pd.concat([signal, forward_return], axis=1, join="inner").dropna()
    aligned.columns = ["signal", "forward_return"]

    scorer = information_coefficient if method == "pearson" else rank_information_coefficient
    values = []
    for i in range(len(aligned)):
        lo = max(0, i + 1 - window)
        window_slice = aligned.iloc[lo : i + 1]
        if len(window_slice) < min_periods:
            values.append(float("nan"))
        else:
            values.append(scorer(window_slice["signal"], window_slice["forward_return"]))
    return pd.Series(values, index=aligned.index, name=f"rolling_{method}_ic")


def decay_curve(
    signal: pd.Series,
    prices: pd.Series,
    horizons: Sequence[int],
    method: str = "pearson",
) -> pd.Series:
    """IC (or Rank IC) of ``signal`` against the forward return realized
    each of ``horizons`` bars later -- the same starting point in time,
    only the forward-looking window changes.

    This is a different question from ``rolling_ic``: rolling IC asks
    "is this signal still good this month vs. six months ago" (calendar
    time on the x-axis, horizon held fixed). Decay curve asks "how many
    bars can I wait before acting on one signal reading before it's gone
    stale" (horizon on the x-axis, calendar window held fixed -- the full
    aligned sample, at every horizon).

    Returns a Series indexed by ``horizons`` (in the order given).
    """
    if method not in ("pearson", "spearman"):
        raise ValueError("method must be 'pearson' or 'spearman'")
    if any(h <= 0 for h in horizons):
        raise ValueError("all horizons must be positive")

    scorer = information_coefficient if method == "pearson" else rank_information_coefficient
    values = [scorer(signal, prices.shift(-h) / prices - 1.0) for h in horizons]
    return pd.Series(values, index=list(horizons), name=f"decay_curve_{method}_ic")


def icir(ic_series: pd.Series) -> float:
    """Information Coefficient Information Ratio: mean IC / std(IC).

    Measures consistency, not just average strength -- two signals with the
    same mean IC can have very different ICIR if one swings wildly period to
    period while the other holds steady. NaN values in ``ic_series`` are
    dropped before computing. Returns NaN if fewer than 2 valid observations
    remain, or if the IC series has zero standard deviation.
    """
    clean = ic_series.dropna()
    if len(clean) < 2:
        return float("nan")
    std = clean.std(ddof=1)
    # a "constant" IC series can still show a tiny nonzero float std from
    # rounding (e.g. mean(0.1, 0.1, 0.1) isn't bit-identical to 0.1), so use
    # a tolerance rather than exact equality -- otherwise that noise turns
    # into a nonsensical, enormous ICIR instead of the intended NaN.
    if np.isclose(std, 0.0, atol=1e-12):
        return float("nan")
    return float(clean.mean() / std)
