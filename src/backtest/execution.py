"""Rebalance calendar and execution timing.

Signal at close of row i (uses data <= i). Trade at close of row i+lag. New weights earn the
returns of rows i+lag+1 onward. The trading cost is deducted from the return of row i+lag.
"""
import pandas as pd


def signal_positions(index: pd.DatetimeIndex, frequency: str, first: int) -> list[int]:
    """Positions of the last trading day of each week/month/quarter, from position `first` on."""
    freq = {"weekly": "W", "monthly": "M", "quarterly": "Q"}[frequency]
    period = index.to_period(freq)
    last = pd.Series(range(len(index)), index=index).groupby(period).max().values
    return [int(p) for p in last if p >= first and p + 1 < len(index)]
