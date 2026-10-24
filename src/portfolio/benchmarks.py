"""Target-weight rules. Each takes only PAST data (history up to and including the signal date)."""
import numpy as np
import pandas as pd

from src.portfolio import optimization as opt
from src.portfolio import volatility_targeting as vt


def equal_weight(history: pd.DataFrame, **_) -> np.ndarray:
    n = history.shape[1]
    return np.full(n, 1.0 / n)


def minimum_variance(history: pd.DataFrame, cov_window: int, max_weight: float, **_) -> np.ndarray:
    return opt.min_variance(opt.shrunk_cov(history.iloc[-cov_window:]), max_weight)


def mean_variance(history: pd.DataFrame, cov_window: int, max_weight: float, risk_aversion: float,
                  shrinkage: float, **_) -> np.ndarray:
    win = history.iloc[-cov_window:]
    mu_raw = win.mean().values
    mu = shrinkage * mu_raw.mean() + (1 - shrinkage) * mu_raw   # shrink toward the grand mean
    return opt.mean_variance(mu, opt.shrunk_cov(win), risk_aversion, max_weight)


def vol_adaptive(history: pd.DataFrame, forecast_vol: float, target_vol: float,
                 min_exposure: float = 0.0, max_exposure: float = 1.0, **_) -> np.ndarray:
    """Equal-weighted risky sleeve scaled by target_vol/forecast_vol; remainder is cash (RF)."""
    n = history.shape[1]
    return vt.exposure(forecast_vol, target_vol, min_exposure, max_exposure) * np.full(n, 1.0 / n)
