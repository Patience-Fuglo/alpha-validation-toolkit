import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from quant_toolkit.metrics.factor_neutralization import OLSResult, fama_french_alpha, ols_regression


def test_ols_recovers_known_coefficients():
    rng = np.random.default_rng(1)
    n = 500
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    noise = rng.normal(scale=0.01, size=n)
    y = 2.0 + 3.0 * x1 - 1.0 * x2 + noise

    X = pd.DataFrame({"x1": x1, "x2": x2})
    result = ols_regression(pd.Series(y), X)

    assert result.alpha == pytest.approx(2.0, abs=0.01)
    assert result.betas["x1"] == pytest.approx(3.0, abs=0.01)
    assert result.betas["x2"] == pytest.approx(-1.0, abs=0.01)
    assert result.r_squared > 0.99


def test_ols_matches_sklearn_reference_implementation():
    rng = np.random.default_rng(2)
    n = 300
    X_raw = rng.normal(size=(n, 3))
    y_raw = X_raw @ np.array([0.5, -0.2, 1.1]) + 0.3 + rng.normal(scale=0.5, size=n)

    X = pd.DataFrame(X_raw, columns=["a", "b", "c"])
    y = pd.Series(y_raw)

    ours = ols_regression(y, X)

    sk = LinearRegression().fit(X_raw, y_raw)
    assert ours.alpha == pytest.approx(sk.intercept_, abs=1e-8)
    for i, col in enumerate(["a", "b", "c"]):
        assert ours.betas[col] == pytest.approx(sk.coef_[i], abs=1e-8)
    assert ours.r_squared == pytest.approx(sk.score(X_raw, y_raw), abs=1e-8)


def test_ols_unrelated_data_gives_near_zero_beta_and_low_r_squared():
    rng = np.random.default_rng(3)
    n = 1000
    x = rng.normal(size=n)
    y = rng.normal(size=n)  # genuinely unrelated to x

    result = ols_regression(pd.Series(y), pd.DataFrame({"x": x}))
    assert abs(result.betas["x"]) < 0.15
    assert result.r_squared < 0.02
    assert not result.is_significant("x")


def test_ols_aligns_by_index_and_drops_nan():
    y = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0], index=range(8))
    x = pd.Series([1.0, 2.0, 3.0, np.nan, 5.0, 6.0, 7.0, 8.0], index=range(8))
    # only indices 0,1,4,5,6,7 have both sides present
    result = ols_regression(y, pd.DataFrame({"x": x}))
    assert result.n_obs == 6


def test_ols_raises_with_too_few_observations():
    y = pd.Series([1.0, 2.0, 3.0])
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [2.0, 1.0, 4.0]})
    with pytest.raises(ValueError):
        ols_regression(y, X)  # 3 obs, 3 params (intercept + a + b) -> not enough


def test_is_significant_threshold():
    result = OLSResult(
        alpha=0.001,
        betas={"mkt_rf": 0.02, "smb": 2.5},
        r_squared=0.5,
        t_stat_alpha=0.3,
        t_stats={"mkt_rf": 0.1, "smb": 3.2},
        n_obs=500,
    )
    assert not result.is_significant("alpha")
    assert not result.is_significant("mkt_rf")
    assert result.is_significant("smb")


def test_fama_french_alpha_requires_expected_columns():
    returns = pd.Series([0.01, 0.02, 0.03])
    bad_factors = pd.DataFrame({"mkt_rf": [0.01, 0.02, 0.03]})
    with pytest.raises(ValueError):
        fama_french_alpha(returns, bad_factors)


def test_fama_french_alpha_subtracts_risk_free_before_regressing():
    rng = np.random.default_rng(4)
    n = 400
    mkt_rf = rng.normal(scale=0.01, size=n)
    smb = rng.normal(scale=0.01, size=n)
    hml = rng.normal(scale=0.01, size=n)
    rf = np.full(n, 0.0001)
    # returns built as risk-free + a clean market-beta relationship, no alpha
    raw_returns = rf + 1.2 * mkt_rf + 0.0 * smb + 0.0 * hml

    factors = pd.DataFrame({"mkt_rf": mkt_rf, "smb": smb, "hml": hml, "rf": rf})
    result = fama_french_alpha(pd.Series(raw_returns), factors)

    assert result.betas["mkt_rf"] == pytest.approx(1.2, abs=1e-6)
    assert result.alpha == pytest.approx(0.0, abs=1e-6)

    # sanity: regressing the RAW (non-excess) return on the same factors
    # should NOT reproduce the same intercept, confirming the RF subtraction
    # inside fama_french_alpha actually did something
    naive = ols_regression(pd.Series(raw_returns), factors[["mkt_rf", "smb", "hml"]])
    assert naive.alpha != pytest.approx(result.alpha, abs=1e-9)
