import numpy as np
import pandas as pd
import pytest

from src.portfolio import benchmarks as B
from src.portfolio import optimization as opt
from src.portfolio.constraints import check_weights
from src.portfolio.volatility_targeting import exposure


def _hist(n=800, k=6, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.normal(0.0004, 0.01, (n, k)) * np.linspace(0.8, 1.6, k))


def test_min_variance_constraints():
    h = _hist()
    w = B.minimum_variance(h, cov_window=756, max_weight=0.30)
    check_weights(w, 0.30)
    assert np.isclose(w.sum(), 1.0, atol=1e-6) and w.min() >= -1e-9


def test_mean_variance_constraints():
    w = B.mean_variance(_hist(), cov_window=756, max_weight=0.35, risk_aversion=5, shrinkage=0.9)
    check_weights(w, 0.35)
    assert np.isclose(w.sum(), 1.0, atol=1e-6)


def test_infeasible_cap_raises():
    with pytest.raises(ValueError):
        opt.min_variance(np.eye(3) * 1e-4, max_weight=0.2)


def test_check_weights_rejects_violations():
    with pytest.raises(ValueError):
        check_weights(np.array([-0.1, 1.1]), 1.5)
    with pytest.raises(ValueError):
        check_weights(np.array([0.6, 0.6]))          # leverage
    with pytest.raises(ValueError):
        check_weights(np.array([0.5, 0.5]), max_weight=0.4)


def test_vol_target_exposure_no_leverage():
    assert exposure(0.20, 0.10) == 0.5
    assert exposure(0.05, 0.10) == 1.0                # capped: no leverage
    with pytest.raises(ValueError):
        exposure(float("nan"), 0.1)
