"""Forecast accuracy metrics and a Diebold-Mariano test robust to overlapping horizons."""
import numpy as np
import pandas as pd
from scipy import stats


def qlike(actual_vol: pd.Series, pred_vol: pd.Series) -> pd.Series:
    """QLIKE on variances: s/h - ln(s/h) - 1 with s = actual^2, h = forecast^2 (0 iff perfect)."""
    ratio = (actual_vol ** 2) / (pred_vol ** 2)
    return ratio - np.log(ratio) - 1.0


def summarize(actual: pd.Series, preds: pd.DataFrame) -> pd.DataFrame:
    rows = {}
    for m in preds:
        e = preds[m] - actual
        rows[m] = {"n": int(e.notna().sum()), "RMSE": float(np.sqrt((e ** 2).mean())),
                   "MAE": float(e.abs().mean()), "QLIKE": float(qlike(actual, preds[m]).mean()),
                   "bias": float(e.mean())}
    return pd.DataFrame(rows).T


def diebold_mariano(loss_a: pd.Series, loss_b: pd.Series, horizon: int) -> dict:
    """H0: equal expected loss. d = loss_a - loss_b; d<0 favours model A.

    Long-run variance uses Newey-West with bandwidth horizon-1 (overlap of h-step targets)
    plus a Harvey-Leybourne-Newbold small-sample correction.
    """
    d = (loss_a - loss_b).dropna().values
    n = len(d)
    dbar = d.mean()
    lag = max(horizon - 1, 0)
    dc = d - dbar
    lrv = (dc @ dc) / n
    for k in range(1, lag + 1):
        w = 1 - k / (lag + 1)
        lrv += 2 * w * (dc[k:] @ dc[:-k]) / n
    stat = dbar / np.sqrt(lrv / n)
    hln = np.sqrt((n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n)
    stat *= hln
    p = 2 * (1 - stats.t.cdf(abs(stat), df=n - 1))
    return {"mean_loss_diff": float(dbar), "dm_stat": float(stat), "p_value": float(p), "n": n}
