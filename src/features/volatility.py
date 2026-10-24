"""Volatility estimators and the forward-looking realized-volatility target."""
import numpy as np
import pandas as pd


def annualize(daily_vol, periods: int = 252):
    return daily_vol * np.sqrt(periods)


def rolling_rms_vol(r: pd.Series, window: int, periods: int = 252) -> pd.Series:
    """Backward-looking annualised root-mean-square volatility over r[t-window+1..t]."""
    return annualize(np.sqrt((r ** 2).rolling(window).mean()), periods)


def rolling_std_vol(r: pd.Series, window: int, periods: int = 252) -> pd.Series:
    return annualize(r.rolling(window).std(), periods)


def forward_realized_vol(r: pd.Series, horizon: int, periods: int = 252) -> pd.Series:
    """Target at date t: annualised RMS of r[t+1..t+horizon]. Uses ONLY returns after t.

    Zero-mean realized volatility, sqrt(mean(r^2)), is the standard realized-volatility
    definition and is directly comparable with GARCH variance forecasts. Last `horizon`
    rows are NaN because the window is not complete.
    """
    fwd_ms = (r ** 2)[::-1].rolling(horizon).mean()[::-1].shift(-1)
    return annualize(np.sqrt(fwd_ms), periods).rename(f"rv_fwd_{horizon}")
