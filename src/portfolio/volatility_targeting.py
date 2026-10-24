"""Volatility-adaptive exposure rule."""
import numpy as np


def exposure(forecast_vol: float, target_vol: float, min_exposure: float = 0.0, max_exposure: float = 1.0) -> float:
    """Share of capital in the risky sleeve: target/forecast, clipped (max_exposure=1 means no leverage)."""
    if not np.isfinite(forecast_vol) or forecast_vol <= 0:
        raise ValueError("invalid volatility forecast")
    return float(np.clip(target_vol / forecast_vol, min_exposure, max_exposure))
