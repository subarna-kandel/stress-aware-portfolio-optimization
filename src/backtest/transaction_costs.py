"""Turnover and proportional transaction costs."""
import numpy as np


def turnover(w_before: np.ndarray, w_after: np.ndarray) -> float:
    """One-way-sum turnover: sum |w_after - w_before| over risky assets (cash trades are free)."""
    return float(np.abs(np.asarray(w_after) - np.asarray(w_before)).sum())


def cost(turn: float, bps: float) -> float:
    """Cost as a fraction of portfolio value: bps is charged per unit of traded notional."""
    return turn * bps / 1e4
