"""Purged walk-forward cross-validation with embargo, built from scratch.

A plain expanding-window walk-forward split (e.g. ``sklearn.TimeSeriesSplit``)
keeps train strictly before test in time, but that alone is not enough for
financial labels: if a label at time ``t`` is computed from a forward-looking
window (e.g. "return over the next 5 days"), the last few rows of "train"
carry information that overlaps the test period. Two separate leaks follow:

1. **Label overlap at the train/test boundary** — the last ``label_horizon``
   rows before test start have labels that peek into the test window.
   Fixed by *purging* them out of train.
2. **Serial correlation across fold boundaries** — once a fold's test period
   ends, its return/volatility autocorrelation doesn't vanish immediately,
   so the following fold's expanding train window can still be indirectly
   informed by the just-seen test period. Fixed by *embargoing* a buffer
   of rows after every prior test fold before they're eligible for training.

This module implements both from first principles on integer positions,
then maps back to the caller's index.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import pandas as pd


@dataclass
class PurgedWalkForward:
    """Expanding-window walk-forward splitter with label-purging and embargo.

    Parameters
    ----------
    n_splits : int
        Number of walk-forward test folds (>= 1).
    label_horizon : int
        Number of bars a label at position ``t`` looks forward (e.g. a
        5-day-forward-return label -> ``label_horizon=5``). Training rows
        whose label window would overlap the test fold are purged.
    embargo_pct : float
        Fraction of the total sample size to embargo immediately after
        *every* prior test fold's end, before those rows may re-enter a
        later fold's training set. Must be in [0, 1).
    """

    n_splits: int
    label_horizon: int = 0
    embargo_pct: float = 0.0

    def __post_init__(self) -> None:
        if self.n_splits < 1:
            raise ValueError("n_splits must be >= 1")
        if self.label_horizon < 0:
            raise ValueError("label_horizon must be >= 0")
        if not (0.0 <= self.embargo_pct < 1.0):
            raise ValueError("embargo_pct must be in [0, 1)")

    def split(self, index: pd.DatetimeIndex) -> Iterator[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
        """Yield (train_index, test_index) pairs, both chronologically-ordered
        subsets of ``index``, for each of ``n_splits`` folds.
        """
        index = pd.DatetimeIndex(index).sort_values()
        n = len(index)
        n_blocks = self.n_splits + 1
        if n < n_blocks * max(1, self.label_horizon + 1):
            raise ValueError(
                f"Not enough rows ({n}) for {self.n_splits} folds with "
                f"label_horizon={self.label_horizon}; need more history or fewer splits."
            )

        block_bounds = self._block_bounds(n, n_blocks)
        embargo_size = int(self.embargo_pct * n)

        # block_bounds[0] is the seed training block; blocks[1:] are test folds.
        prior_test_ends: list[int] = []
        for fold in range(1, n_blocks):
            test_start, test_end = block_bounds[fold]

            purge_start = max(0, test_start - self.label_horizon)
            train_mask = [True] * purge_start + [False] * (n - purge_start)

            for prior_end in prior_test_ends:
                embargo_end = min(n, prior_end + embargo_size)
                for pos in range(prior_end, embargo_end):
                    if pos < len(train_mask):
                        train_mask[pos] = False

            train_positions = [p for p in range(test_start) if train_mask[p]]
            if not train_positions:
                raise ValueError(
                    f"Fold {fold}: purge/embargo removed all training rows; "
                    "reduce label_horizon, embargo_pct, or n_splits."
                )

            train_idx = index[train_positions]
            test_idx = index[test_start:test_end]
            yield train_idx, test_idx

            prior_test_ends.append(test_end)

    @staticmethod
    def _block_bounds(n: int, n_blocks: int) -> list[tuple[int, int]]:
        """Split ``range(n)`` into ``n_blocks`` contiguous, near-equal blocks."""
        base, remainder = divmod(n, n_blocks)
        bounds = []
        start = 0
        for i in range(n_blocks):
            size = base + (1 if i < remainder else 0)
            bounds.append((start, start + size))
            start += size
        return bounds
