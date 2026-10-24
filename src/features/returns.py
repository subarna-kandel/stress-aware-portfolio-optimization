"""Return helpers. All returns are simple daily decimals (0.01 = 1%)."""
import pandas as pd


def equal_weight_return(asset_returns: pd.DataFrame) -> pd.Series:
    """Daily-rebalanced equal-weighted return of the risky universe (the volatility-forecast target series)."""
    return asset_returns.mean(axis=1).rename("ew_return")


def cumulative_return(r: pd.Series, window: int) -> pd.Series:
    """Compounded return over the trailing `window` observations (uses only r[t-window+1..t])."""
    return (1.0 + r).rolling(window).apply(lambda x: x.prod(), raw=True) - 1.0


def wealth_index(r: pd.Series) -> pd.Series:
    return (1.0 + r).cumprod()


def drawdown(r: pd.Series) -> pd.Series:
    w = wealth_index(r)
    return w / w.cummax() - 1.0
