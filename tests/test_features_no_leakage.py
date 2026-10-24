"""Temporal-leakage tests: features at t must not change when the future is altered."""
import numpy as np
import pandas as pd

from src.features.build_features import build_dataset
from src.features.volatility import forward_realized_vol


def _series(n=400, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=n)
    r = pd.Series(rng.normal(0, 0.01, n), index=idx)
    vix = pd.Series(15 + rng.normal(0, 2, n).cumsum() * 0.1 + 10, index=idx).clip(lower=9)
    return r, vix


def test_features_ignore_future():
    r, vix = _series()
    t = 300
    base = build_dataset(r, vix)
    r2, vix2 = r.copy(), vix.copy()
    r2.iloc[t + 1:] = r2.iloc[t + 1:] * 5 + 0.05
    vix2.iloc[t + 1:] = vix2.iloc[t + 1:] * 3
    alt = build_dataset(r2, vix2)
    feats = [c for c in base.columns if c != "target"]
    pd.testing.assert_frame_equal(base[feats].iloc[: t + 1], alt[feats].iloc[: t + 1])


def test_target_uses_only_future_returns():
    r, _ = _series()
    t, h = 200, 5
    base = forward_realized_vol(r, h)
    r2 = r.copy()
    r2.iloc[: t + 1] = 0.123          # change everything up to and including t
    alt = forward_realized_vol(r2, h)
    assert np.isclose(base.iloc[t], alt.iloc[t])
    expected = np.sqrt((r.iloc[t + 1: t + 1 + h] ** 2).mean()) * np.sqrt(252)
    assert np.isclose(base.iloc[t], expected)
    assert base.iloc[-h:].isna().all()


def test_target_not_in_features():
    r, vix = _series()
    d = build_dataset(r, vix)
    assert "target" not in [c for c in d.columns if c != "target"]
    assert not any(c.startswith("rv_fwd") for c in d.columns)
