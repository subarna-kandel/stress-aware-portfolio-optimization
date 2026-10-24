"""Model B: GARCH(1,1), re-estimated on a rolling window; forecasts filtered daily with fixed parameters."""
import logging
import warnings

import numpy as np
import pandas as pd
from arch import arch_model

log = logging.getLogger(__name__)


def _spec(returns_pct: pd.Series, p: int, q: int, dist: str):
    return arch_model(returns_pct, mean="Constant", vol="GARCH", p=p, q=q, dist=dist, rescale=False)


def walk_forward_garch(r: pd.Series, origins: list[int], train_window: int, refit_every: int,
                       horizon: int = 5, p: int = 1, q: int = 1, dist: str = "normal",
                       periods: int = 252) -> tuple[pd.Series, pd.DataFrame]:
    """origins: integer positions in r at which forecasts are produced (sorted, consecutive).

    Parameters are estimated every `refit_every` origins using r[origin-train_window+1 .. origin]
    only. For the origins in that block the conditional variance is filtered with those fixed
    parameters; the filter at origin t uses r[<=t] only.
    """
    rp = r * 100.0
    out = pd.Series(np.nan, index=r.index, name="garch")
    fit_log = []
    params_prev = None
    for b in range(0, len(origins), refit_every):
        block = origins[b:b + refit_every]
        t0 = block[0]
        start = t0 - train_window + 1
        train = rp.iloc[start:t0 + 1]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = _spec(train, p, q, dist).fit(disp="off", show_warning=False)
        ok = res.convergence_flag == 0 and np.all(np.isfinite(res.params.values))
        if not ok:
            log.warning("GARCH convergence flag %s at %s", res.convergence_flag, r.index[t0])
        fit_log.append({"origin_date": r.index[t0], "convergence_flag": int(res.convergence_flag),
                        "used_previous_params": (not ok) and params_prev is not None,
                        **{k: float(v) for k, v in res.params.items()}})
        params = res.params if ok else params_prev
        if params is None:
            params = res.params  # first fit failed: keep its estimates, flagged in fit_log
        params_prev = params
        seg = rp.iloc[start:block[-1] + 1]
        fixed = _spec(seg, p, q, dist).fix(params)
        first = block[0] - start
        fc = fixed.forecast(horizon=horizon, start=first, reindex=False)
        var = fc.variance.values  # (n_origins, horizon), %^2 per day
        vol = np.sqrt(var.mean(axis=1)) / 100.0 * np.sqrt(periods)
        out.iloc[block] = vol[: len(block)]
    return out, pd.DataFrame(fit_log)
