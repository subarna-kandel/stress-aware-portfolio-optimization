"""Optional Model D: gradient-boosted trees on log realized volatility (same walk-forward protocol as the RF)."""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

from src.models.random_forest import training_rows


def walk_forward_xgb(data: pd.DataFrame, feature_cols: list[str], origins: list[int], train_window: int,
                     refit_every: int, horizon: int, cfg: dict, seed: int = 42):
    X_all = data[feature_cols].values
    y_all = np.log(data["target"].clip(lower=1e-4)).values
    out = pd.Series(np.nan, index=data.index, name="xgboost")
    log, importances, best = [], [], None
    base_kw = dict(n_estimators=cfg["n_estimators"], learning_rate=cfg["learning_rate"], subsample=0.8,
                   colsample_bytree=0.8, random_state=seed, n_jobs=cfg["n_jobs"], tree_method="hist")
    for k, b in enumerate(range(0, len(origins), refit_every)):
        block = origins[b:b + refit_every]
        t0 = block[0]
        rows = training_rows(t0, train_window, horizon)
        rows = rows[np.isfinite(y_all[rows]) & np.isfinite(X_all[rows]).all(axis=1)]
        Xtr, ytr = X_all[rows], y_all[rows]
        if best is None or k % cfg["tune_every_n_refits"] == 0:
            grid = {"max_depth": cfg["max_depth_grid"], "min_child_weight": cfg["min_child_weight_grid"]}
            gs = GridSearchCV(xgb.XGBRegressor(**base_kw), grid, cv=TimeSeriesSplit(cfg["cv_splits"], gap=horizon),
                              scoring="neg_mean_squared_error", n_jobs=1).fit(Xtr, ytr)
            best = gs.best_params_
        model = xgb.XGBRegressor(**base_kw, **best).fit(Xtr, ytr)
        # smearing factor from in-sample residuals of the final model is too optimistic for boosting; use the
        # residuals of the last time-series CV fold instead (training data only)
        cut = int(len(rows) * 0.8)
        m2 = xgb.XGBRegressor(**base_kw, **best).fit(Xtr[:cut - horizon], ytr[:cut - horizon])
        smear = float(np.exp(0.5 * np.var(ytr[cut:] - m2.predict(Xtr[cut:]))))
        out.iloc[block] = np.exp(model.predict(X_all[block])) * smear
        log.append({"origin_date": data.index[t0], "n_train": len(rows), "smear": smear, **best})
        importances.append(dict(zip(feature_cols, model.feature_importances_)))
    return out, pd.DataFrame(log), pd.DataFrame(importances).mean().sort_values(ascending=False)
