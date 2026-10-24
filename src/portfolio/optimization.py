"""Covariance estimation and constrained optimisers (long-only, fully invested, weight cap)."""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf


def shrunk_cov(returns: pd.DataFrame) -> np.ndarray:
    """Ledoit-Wolf covariance of daily returns, computed on the supplied (past-only) window."""
    return LedoitWolf().fit(returns.values).covariance_


def _solve(objective, n: int, max_weight: float, w0=None) -> np.ndarray:
    if max_weight * n < 1 - 1e-12:
        raise ValueError("max_weight too small to be fully invested")
    w0 = np.full(n, 1.0 / n) if w0 is None else w0
    res = minimize(objective, w0, method="SLSQP", bounds=[(0.0, max_weight)] * n,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
                   options={"maxiter": 500, "ftol": 1e-12})
    if not res.success:
        raise RuntimeError(f"optimiser failed: {res.message}")
    return np.clip(res.x, 0.0, max_weight)


def min_variance(cov: np.ndarray, max_weight: float) -> np.ndarray:
    scale = 1e4  # daily covariances are ~1e-4; rescale for numerical conditioning
    return _solve(lambda w: scale * w @ cov @ w, len(cov), max_weight)


def mean_variance(mu: np.ndarray, cov: np.ndarray, risk_aversion: float, max_weight: float) -> np.ndarray:
    """maximise mu'w - (lambda/2) w'Sigma w, all in daily units."""
    scale = 1e4
    return _solve(lambda w: -scale * (mu @ w - 0.5 * risk_aversion * w @ cov @ w), len(mu), max_weight)
