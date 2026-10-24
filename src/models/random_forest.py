"""Model C: Random Forest regression of log realized volatility, walk-forward with embargoed labels."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit


def training_rows(origin: int, train_window: int, horizon: int) -> np.ndarray:
    """Row positions usable for fitting at forecast origin `origin`.

    A row i has a label spanning returns i+1..i+horizon, so it is only observed at the origin
    if i + horizon <= origin. This excludes labels that reach into the test period.
    """
    last = origin - horizon
    first = last - train_window + 1
    return np.arange(max(first, 0), last + 1)


def walk_forward_rf(data: pd.DataFrame, feature_cols: list[str], origins: list[int], train_window: int,
                    refit_every: int, horizon: int, cfg: dict, seed: int = 42) -> tuple[pd.Series, pd.DataFrame, pd.Series]:
    X_all = data[feature_cols].values
    y_all = np.log(data["target"].clip(lower=1e-4)).values
    out = pd.Series(np.nan, index=data.index, name="random_forest")
    log, importances = [], []
    best = None
    for k, b in enumerate(range(0, len(origins), refit_every)):
        block = origins[b:b + refit_every]
        t0 = block[0]
        rows = training_rows(t0, train_window, horizon)
        rows = rows[np.isfinite(y_all[rows]) & np.isfinite(X_all[rows]).all(axis=1)]
        Xtr, ytr = X_all[rows], y_all[rows]
        if best is None or k % cfg["tune_every_n_refits"] == 0:
            grid = {"min_samples_leaf": cfg["min_samples_leaf_grid"], "max_depth": cfg["max_depth_grid"]}
            base = RandomForestRegressor(n_estimators=100, max_features=cfg["max_features"],
                                         random_state=seed, n_jobs=cfg["n_jobs"])
            cv = TimeSeriesSplit(n_splits=cfg["cv_splits"], gap=horizon)
            gs = GridSearchCV(base, grid, cv=cv, scoring="neg_mean_squared_error", n_jobs=1).fit(Xtr, ytr)
            best = gs.best_params_
        model = RandomForestRegressor(n_estimators=cfg["n_estimators"], max_features=cfg["max_features"],
                                      random_state=seed, n_jobs=cfg["n_jobs"], oob_score=True, **best).fit(Xtr, ytr)
        # log-model retransformation: smearing factor from out-of-bag residuals (training data only)
        smear = float(np.exp(0.5 * np.var(ytr - model.oob_prediction_)))
        pred = model.predict(X_all[block])
        out.iloc[block] = np.exp(pred) * smear
        log.append({"origin_date": data.index[t0], "train_start": data.index[rows[0]],
                    "train_end": data.index[rows[-1]], "n_train": len(rows), "smear": smear, **best})
        importances.append(dict(zip(feature_cols, model.feature_importances_)))
    imp = pd.DataFrame(importances).mean().sort_values(ascending=False)
    return out, pd.DataFrame(log), imp
