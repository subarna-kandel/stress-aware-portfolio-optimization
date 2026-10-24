"""Market-stress labels. Labels for date t depend only on information up to t-1."""
import pandas as pd


def stress_flag(vix_close: pd.Series, threshold: float) -> pd.Series:
    """True on day t if VIX at the close of t-1 was >= threshold (known before day t's return)."""
    return (vix_close.shift(1) >= threshold).rename("stress_flag")
