"""Feature matrix for volatility forecasting.

Row t = information available at the close of day t. Every feature at row t is a function of
r[<=t] and vix[<=t] only; the target at row t is a function of r[>t] only.
"""
import numpy as np
import pandas as pd

from src.features import returns as R
from src.features import volatility as V


def build_features(r: pd.Series, vix: pd.Series, windows=(5, 21, 63), vix_lags=(1, 5),
                   horizon: int = 5, periods: int = 252) -> pd.DataFrame:
    X = pd.DataFrame(index=r.index)
    for k in range(5):
        X[f"ret_lag{k}"] = r.shift(k)
    for w in windows:
        X[f"cumret_{w}"] = R.cumulative_return(r, w)
        X[f"vol_{w}"] = V.rolling_rms_vol(r, w, periods)
    X["abs_ret"] = r.abs()
    X["vix_level"] = vix
    for k in vix_lags:
        X[f"vix_chg_{k}"] = np.log(vix / vix.shift(k))
    wealth = R.wealth_index(r)
    X["drawdown_252"] = wealth / wealth.rolling(252, min_periods=63).max() - 1.0
    X["vix_vol_ratio"] = np.log(vix / 100.0) - np.log(X["vol_21"])
    return X


def build_dataset(r: pd.Series, vix: pd.Series, horizon: int = 5, **kw) -> pd.DataFrame:
    X = build_features(r, vix, horizon=horizon, **kw)
    X["target"] = V.forward_realized_vol(r, horizon)
    return X
