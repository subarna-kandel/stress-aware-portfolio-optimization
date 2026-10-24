"""Model A: rolling historical-volatility baseline."""
import pandas as pd

from src.features.volatility import rolling_rms_vol


def forecast(r: pd.Series, window: int = 21, periods: int = 252) -> pd.Series:
    """Forecast made at close of t = trailing RMS volatility over r[t-window+1..t]."""
    return rolling_rms_vol(r, window, periods).rename("historical_vol")
