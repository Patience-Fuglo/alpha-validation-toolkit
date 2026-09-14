import pandas as pd
import pytest

from quant_toolkit.validation import PurgedWalkForward


def business_days(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2020-01-01", periods=n, freq="B")


def test_yields_exactly_n_splits_folds():
    idx = business_days(200)
    splitter = PurgedWalkForward(n_splits=5, label_horizon=3, embargo_pct=0.01)
    folds = list(splitter.split(idx))
    assert len(folds) == 5


def test_train_always_precedes_test_chronologically():
    idx = business_days(200)
    splitter = PurgedWalkForward(n_splits=5, label_horizon=3, embargo_pct=0.01)
    for train_idx, test_idx in splitter.split(idx):
        assert train_idx.max() < test_idx.min()


def test_train_and_test_never_overlap():
    idx = business_days(200)
    splitter = PurgedWalkForward(n_splits=5, label_horizon=3, embargo_pct=0.01)
    for train_idx, test_idx in splitter.split(idx):
        assert len(train_idx.intersection(test_idx)) == 0


def test_purge_drops_label_horizon_rows_before_test():
    idx = business_days(200)
    label_horizon = 5
    splitter = PurgedWalkForward(n_splits=4, label_horizon=label_horizon, embargo_pct=0.0)
    for train_idx, test_idx in splitter.split(idx):
        test_start_pos = idx.get_loc(test_idx.min())
        purged_zone = idx[max(0, test_start_pos - label_horizon):test_start_pos]
        assert len(train_idx.intersection(purged_zone)) == 0


def test_embargo_excludes_rows_after_prior_test_fold():
    idx = business_days(300)
    embargo_pct = 0.05
    splitter = PurgedWalkForward(n_splits=5, label_horizon=0, embargo_pct=embargo_pct)
    embargo_size = int(embargo_pct * len(idx))

    folds = list(splitter.split(idx))
    prior_test_ends = []
    for train_idx, test_idx in folds:
        for prior_end_idx in prior_test_ends:
            # embargo covers the buffer strictly *after* the prior fold's last
            # test day -- that last day itself is legitimately reusable as
            # training data once its fold is over.
            end_pos = idx.get_loc(prior_end_idx) + 1
            embargo_zone = idx[end_pos:end_pos + embargo_size]
            assert len(train_idx.intersection(embargo_zone)) == 0
        prior_test_ends.append(test_idx.max())


def test_zero_embargo_and_horizon_reduces_to_plain_expanding_walk_forward():
    idx = business_days(120)
    splitter = PurgedWalkForward(n_splits=3, label_horizon=0, embargo_pct=0.0)
    folds = list(splitter.split(idx))
    train0, test0 = folds[0]
    assert train0.max() == idx[idx.get_loc(test0.min()) - 1]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_splits": 0},
        {"n_splits": 3, "label_horizon": -1},
        {"n_splits": 3, "embargo_pct": 1.0},
        {"n_splits": 3, "embargo_pct": -0.1},
    ],
)
def test_invalid_params_raise(kwargs):
    with pytest.raises(ValueError):
        PurgedWalkForward(**kwargs)


def test_insufficient_data_raises():
    idx = business_days(10)
    splitter = PurgedWalkForward(n_splits=5, label_horizon=5, embargo_pct=0.1)
    with pytest.raises(ValueError):
        list(splitter.split(idx))
