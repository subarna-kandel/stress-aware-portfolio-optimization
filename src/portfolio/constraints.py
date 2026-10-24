"""Weight constraints. Weight vectors cover the risky assets only (cash is the residual)."""
import numpy as np


def check_weights(w: np.ndarray, max_weight: float = 1.0, long_only: bool = True,
                  allow_leverage: bool = False, tol: float = 1e-8) -> None:
    w = np.asarray(w, dtype=float)
    if not np.isfinite(w).all():
        raise ValueError("non-finite weights")
    if long_only and (w < -tol).any():
        raise ValueError("short position in a long-only portfolio")
    if (np.abs(w) > max_weight + tol).any():
        raise ValueError(f"weight above max_weight={max_weight}")
    if not allow_leverage and w.sum() > 1 + tol:
        raise ValueError("gross exposure above 100% (leverage)")
