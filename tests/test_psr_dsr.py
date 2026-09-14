import math

import numpy as np
import pytest
from scipy import stats

from quant_toolkit.metrics.psr_dsr import (
    _norm_cdf,
    _norm_ppf,
    _skewness,
    _kurtosis,
    deflated_sharpe_ratio,
    expected_max_sharpe_under_null,
    probabilistic_sharpe_ratio,
    sharpe_ratio,
)


# ---- building blocks: normal CDF/PPF, skew, kurtosis ----------------------


def test_norm_cdf_known_values():
    assert _norm_cdf(0.0) == pytest.approx(0.5, abs=1e-12)
    assert _norm_cdf(1.959964) == pytest.approx(0.975, abs=1e-5)
    assert _norm_cdf(-1.0) == pytest.approx(0.158655, abs=1e-5)


def test_norm_ppf_matches_scipy_reference():
    for p in [0.001, 0.05, 0.25, 0.5, 0.75, 0.95, 0.999]:
        ours = _norm_ppf(p)
        reference = stats.norm.ppf(p)
        assert ours == pytest.approx(reference, abs=1e-6)


def test_norm_ppf_is_inverse_of_norm_cdf():
    for x in [-3.0, -1.0, 0.0, 0.5, 2.0]:
        p = _norm_cdf(x)
        assert _norm_ppf(p) == pytest.approx(x, abs=1e-6)


def test_norm_ppf_rejects_out_of_range_probability():
    with pytest.raises(ValueError):
        _norm_ppf(0.0)
    with pytest.raises(ValueError):
        _norm_ppf(1.0)


def test_skewness_matches_scipy_reference():
    rng = np.random.default_rng(1)
    x = rng.exponential(size=500)  # genuinely skewed, not symmetric
    assert _skewness(x) == pytest.approx(stats.skew(x, bias=True), abs=1e-8)


def test_kurtosis_matches_scipy_reference():
    rng = np.random.default_rng(2)
    x = rng.standard_t(df=3, size=500)  # fat-tailed
    assert _kurtosis(x) == pytest.approx(stats.kurtosis(x, fisher=False, bias=True), abs=1e-8)


def test_kurtosis_of_normal_data_is_near_three():
    rng = np.random.default_rng(3)
    x = rng.normal(size=20000)
    assert _kurtosis(x) == pytest.approx(3.0, abs=0.15)


# ---- sharpe ratio -----------------------------------------------------


def test_sharpe_ratio_known_value():
    x = np.array([0.02, 0.04, 0.00, -0.02, 0.06])
    expected = x.mean() / x.std(ddof=1)
    assert sharpe_ratio(x) == pytest.approx(expected)


def test_sharpe_ratio_zero_variance_returns_nan():
    assert np.isnan(sharpe_ratio(np.array([0.01, 0.01, 0.01])))


# ---- probabilistic sharpe ratio ----------------------------------------


def test_psr_is_half_when_sample_sharpe_equals_benchmark():
    rng = np.random.default_rng(4)
    x = rng.normal(loc=0.001, scale=0.02, size=100)
    sr_hat = sharpe_ratio(x)
    assert probabilistic_sharpe_ratio(x, benchmark_sr=sr_hat) == pytest.approx(0.5, abs=1e-9)


def test_psr_moves_further_from_half_with_more_observations_at_fixed_sharpe():
    # chosen so PSR at n=50 lands comfortably inside (0.5, 1) rather than
    # already saturated at the float boundary -- otherwise "moves further
    # toward 1" has no room left to demonstrate.
    rng = np.random.default_rng(9)
    base = rng.normal(loc=0.002, scale=0.02, size=50)
    assert sharpe_ratio(base) > 0  # sanity-check the premise before testing the property

    small_n = probabilistic_sharpe_ratio(base, benchmark_sr=0.0)
    # tiling repeats the exact same empirical mean/std/skew/kurtosis, only n changes
    large_n = probabilistic_sharpe_ratio(np.tile(base, 10), benchmark_sr=0.0)
    assert 0.5 < small_n < 0.95
    assert large_n > small_n


def test_psr_decreases_as_benchmark_rises():
    rng = np.random.default_rng(6)
    x = rng.normal(loc=0.002, scale=0.02, size=200)
    low_bar = probabilistic_sharpe_ratio(x, benchmark_sr=0.0)
    high_bar = probabilistic_sharpe_ratio(x, benchmark_sr=1.0)
    assert high_bar < low_bar


def test_psr_too_few_observations_returns_nan():
    assert np.isnan(probabilistic_sharpe_ratio(np.array([0.01, 0.02])))


def test_psr_is_valid_probability_or_nan_under_extreme_distributions():
    rng = np.random.default_rng(9)
    extreme_cases = [
        np.concatenate([[50.0], rng.normal(-1, 0.1, size=99)]),  # one huge spike
        rng.standard_cauchy(size=200),  # pathologically heavy tails
        rng.exponential(scale=5.0, size=200) - 5.0,
    ]
    for x in extreme_cases:
        result = probabilistic_sharpe_ratio(x)
        assert np.isnan(result) or 0.0 <= result <= 1.0


# ---- expected max sharpe under the null (multiple-testing threshold) ---


def test_expected_max_sharpe_increases_with_more_trials():
    low = expected_max_sharpe_under_null(n_trials=2, sr_std=0.5)
    high = expected_max_sharpe_under_null(n_trials=50, sr_std=0.5)
    assert high > low


def test_expected_max_sharpe_scales_with_sr_std():
    base = expected_max_sharpe_under_null(n_trials=15, sr_std=0.5)
    doubled = expected_max_sharpe_under_null(n_trials=15, sr_std=1.0)
    assert doubled == pytest.approx(2 * base, abs=1e-9)


def test_expected_max_sharpe_requires_at_least_two_trials():
    with pytest.raises(ValueError):
        expected_max_sharpe_under_null(n_trials=1, sr_std=0.5)


def test_expected_max_sharpe_rejects_negative_sr_std():
    with pytest.raises(ValueError):
        expected_max_sharpe_under_null(n_trials=10, sr_std=-0.1)


# ---- deflated sharpe ratio ----------------------------------------------


def test_dsr_is_at_most_psr_against_zero_benchmark():
    # correcting for having tried multiple trials can only raise the bar,
    # never lower it -- DSR should never exceed PSR(benchmark=0)
    rng = np.random.default_rng(8)
    x = rng.normal(loc=0.0015, scale=0.02, size=300)
    psr_vs_zero = probabilistic_sharpe_ratio(x, benchmark_sr=0.0)
    dsr = deflated_sharpe_ratio(x, n_trials=15)
    assert dsr <= psr_vs_zero + 1e-9


def test_dsr_drops_as_n_trials_grows():
    rng = np.random.default_rng(10)
    x = rng.normal(loc=0.0015, scale=0.02, size=300)
    dsr_few = deflated_sharpe_ratio(x, n_trials=2)
    dsr_many = deflated_sharpe_ratio(x, n_trials=100)
    assert dsr_many <= dsr_few


def test_dsr_accepts_explicit_sr_std():
    rng = np.random.default_rng(11)
    x = rng.normal(loc=0.0015, scale=0.02, size=300)
    result = deflated_sharpe_ratio(x, n_trials=15, sr_std=0.3)
    assert 0.0 <= result <= 1.0
