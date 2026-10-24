"""Performance and risk metrics on daily decimal returns."""
import numpy as np
import pandas as pd


def max_drawdown(r: pd.Series) -> float:
    w = (1 + r).cumprod()
    return float((w / w.cummax() - 1).min())


def ann_return(r: pd.Series, periods: int = 252) -> float:
    """Geometric annualised return."""
    return float((1 + r).prod() ** (periods / len(r)) - 1)


def ann_vol(r: pd.Series, periods: int = 252) -> float:
    return float(r.std() * np.sqrt(periods))


def sharpe(r: pd.Series, rf: pd.Series, periods: int = 252) -> float:
    ex = r - rf.reindex(r.index)
    return float(ex.mean() / ex.std() * np.sqrt(periods))


def sortino(r: pd.Series, rf: pd.Series, periods: int = 252) -> float:
    """Downside deviation = sqrt(mean(min(r - rf, 0)^2)) over ALL days (target = daily risk-free rate)."""
    ex = r - rf.reindex(r.index)
    dd = np.sqrt((np.minimum(ex, 0) ** 2).mean())
    return float(ex.mean() / dd * np.sqrt(periods))


def summarize(res: pd.DataFrame, rf: pd.Series, col: str = "net_return", periods: int = 252) -> dict:
    r = res[col]
    mdd = max_drawdown(r)
    ar = ann_return(r, periods)
    years = len(r) / periods
    return {
        "cumulative_return": float((1 + r).prod() - 1), "ann_return": ar, "ann_vol": ann_vol(r, periods),
        "sharpe": sharpe(r, rf, periods), "sortino": sortino(r, rf, periods), "max_drawdown": mdd,
        "calmar": float(ar / abs(mdd)) if mdd < 0 else np.nan,
        "ann_turnover": float(res["turnover"].sum() / years), "ann_cost": float(res["transaction_cost"].sum() / years),
        "avg_exposure": float(res["exposure"].mean()), "n_days": int(len(r)),
    }


def regime_summary(r: pd.Series, rf: pd.Series, flag: pd.Series, periods: int = 252) -> pd.DataFrame:
    flag = flag.reindex(r.index)
    rows = {}
    for name, mask in [("stress", flag), ("calm", ~flag)]:
        x = r[mask]
        if len(x) < 20:
            continue
        rows[name] = {"n_days": int(len(x)), "ann_mean_return": float(x.mean() * periods),
                      "ann_vol": ann_vol(x, periods), "sharpe": sharpe(x, rf, periods),
                      "worst_day": float(x.min()), "pct_days": float(len(x) / len(r))}
    return pd.DataFrame(rows).T


def stationary_bootstrap_sharpe_diff(a: pd.Series, b: pd.Series, rf: pd.Series, n_boot: int = 2000,
                                     mean_block: int = 20, periods: int = 252, seed: int = 42) -> dict:
    """Politis-Romano stationary bootstrap for Sharpe(a) - Sharpe(b); preserves serial dependence."""
    ea = (a - rf.reindex(a.index)).values
    eb = (b - rf.reindex(b.index)).values
    n = len(ea)
    rng = np.random.default_rng(seed)
    p = 1.0 / mean_block

    def sr(x):
        return x.mean() / x.std() * np.sqrt(periods)

    obs = sr(ea) - sr(eb)
    diffs = np.empty(n_boot)
    for k in range(n_boot):
        starts = rng.integers(0, n, size=n)
        new_block = rng.random(n) < p
        new_block[0] = True
        idx = np.empty(n, dtype=int)
        cur = 0
        for i in range(n):
            cur = starts[i] if new_block[i] else (cur + 1) % n
            idx[i] = cur
        diffs[k] = sr(ea[idx]) - sr(eb[idx])
    centered = diffs - diffs.mean()
    pval = float((np.abs(centered) >= abs(obs)).mean())
    return {"sharpe_diff": float(obs), "p_value": pval, "ci_low": float(np.percentile(diffs, 2.5)),
            "ci_high": float(np.percentile(diffs, 97.5))}
