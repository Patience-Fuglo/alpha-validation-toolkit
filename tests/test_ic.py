import numpy as np
import pandas as pd
import pytest
from scipy import stats

from quant_toolkit.metrics import icir, information_coefficient, rank_information_coefficient, rolling_ic


def series(values, index=None):
    return pd.Series(values, index=index if index is not None else range(len(values)))


def test_perfect_positive_correlation_is_one():
    x = series([1, 2, 3, 4, 5])
    assert information_coefficient(x, x) == pytest.approx(1.0)
    assert rank_information_coefficient(x, x) == pytest.approx(1.0)


def test_perfect_negative_correlation_is_minus_one():
    x = series([1, 2, 3, 4, 5])
    y = series([5, 4, 3, 2, 1])
    assert information_coefficient(x, y) == pytest.approx(-1.0)
    assert rank_information_coefficient(x, y) == pytest.approx(-1.0)


def test_pearson_matches_pandas_reference_implementation():
    rng = np.random.default_rng(7)
    x = series(rng.normal(size=200))
    y = series(0.3 * x.to_numpy() + rng.normal(size=200))
    ours = information_coefficient(x, y)
    reference = x.corr(y, method="pearson")
    assert ours == pytest.approx(reference, abs=1e-10)


def test_rank_ic_matches_scipy_reference_implementation_with_ties():
    rng = np.random.default_rng(11)
    # force ties by rounding, since real signals often bucket into discrete values
    x = series(np.round(rng.normal(size=150), 1))
    y = series(rng.normal(size=150))
    ours = rank_information_coefficient(x, y)
    reference, _ = stats.spearmanr(x.to_numpy(), y.to_numpy())
    assert ours == pytest.approx(reference, abs=1e-10)


def test_misaligned_index_and_nan_rows_are_dropped():
    x = series([1.0, 2.0, np.nan, 4.0, 5.0], index=[0, 1, 2, 3, 4])
    y = series([10.0, 20.0, 30.0, np.nan, 50.0], index=[0, 1, 2, 3, 4])
    # only indices 0, 1, 4 have both sides non-NaN
    manual = information_coefficient(series([1.0, 2.0, 5.0]), series([10.0, 20.0, 50.0]))
    assert information_coefficient(x, y) == pytest.approx(manual)


def test_disjoint_index_ranges_use_only_overlap():
    x = series([1.0, 2.0, 3.0], index=[0, 1, 2])
    y = series([10.0, 20.0, 30.0], index=[1, 2, 3])
    # overlap is index {1, 2}: x=[2,3], y=[10,20]
    manual = information_coefficient(series([2.0, 3.0]), series([10.0, 20.0]))
    assert information_coefficient(x, y) == pytest.approx(manual)


def test_constant_series_returns_nan():
    x = series([1, 1, 1, 1])
    y = series([1, 2, 3, 4])
    assert np.isnan(information_coefficient(x, y))
    assert np.isnan(rank_information_coefficient(x, y))


def test_fewer_than_two_observations_returns_nan():
    assert np.isnan(information_coefficient(series([1.0]), series([2.0])))


def test_rolling_ic_length_matches_aligned_input():
    x = series(np.arange(20, dtype=float))
    y = series(np.arange(20, dtype=float) * 2)
    result = rolling_ic(x, y, window=5)
    assert len(result) == 20


def test_rolling_ic_nan_before_min_periods():
    x = series(np.arange(10, dtype=float))
    y = series(np.arange(10, dtype=float))
    result = rolling_ic(x, y, window=5)
    assert result.iloc[:4].isna().all()
    assert not result.iloc[4:].isna().any()


def test_rolling_ic_matches_manual_window_computation():
    rng = np.random.default_rng(3)
    x = series(rng.normal(size=30))
    y = series(rng.normal(size=30))
    window = 6
    result = rolling_ic(x, y, window=window)
    manual_last = information_coefficient(x.iloc[-window:], y.iloc[-window:])
    assert result.iloc[-1] == pytest.approx(manual_last)


def test_rolling_ic_invalid_method_raises():
    x = series([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        rolling_ic(x, x, window=2, method="kendall")


def test_rolling_ic_invalid_window_raises():
    x = series([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        rolling_ic(x, x, window=1)


def test_icir_known_values():
    # zero std (constant IC across periods) -> undefined, per documented behavior
    constant = series([0.10, 0.10, 0.10])
    assert np.isnan(icir(constant))

    varying = series([0.05, 0.15, 0.10, 0.20])
    expected = varying.mean() / varying.std(ddof=1)
    assert icir(varying) == pytest.approx(expected)


def test_icir_drops_nan_and_handles_insufficient_data():
    ic_series = series([0.1, np.nan, np.nan])
    assert np.isnan(icir(ic_series))
