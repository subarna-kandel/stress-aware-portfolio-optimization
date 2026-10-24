"""Walk-forward volatility forecasts for the equal-weighted universe return.

Usage: python scripts/run_forecasts.py [--train-window N] [--eval-start YYYY-MM-DD] [--tag name]
"""
import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.config import ROOT, load_config
from src.data.align import load_processed
from src.features import build_features as BF
from src.features.returns import equal_weight_return
from src.models import garch, historical_volatility, random_forest


def run(train_window=None, eval_start=None, tag="main", max_origins=None):
    exp = load_config("experiment")
    W = train_window or exp["training_window_days"]
    h = exp["forecast_horizon_days"]
    ind, aux = load_processed()
    r = equal_weight_return(ind)
    data = BF.build_dataset(r, aux["vix_close"], horizon=h, windows=exp["feature_windows"],
                            vix_lags=exp["vix_lags"], periods=exp["trading_days_per_year"])
    feature_cols = [c for c in data.columns if c != "target"]
    first_valid = int(data[feature_cols].notna().all(axis=1).values.argmax())
    first_origin = first_valid + W + h
    if eval_start is not None:
        first_origin = max(first_origin, int(data.index.searchsorted(pd.Timestamp(eval_start))))
    last_origin = len(data) - 1                      # forecasts are produced even where target is unknown
    origins = list(range(first_origin, last_origin + 1))
    if max_origins:
        origins = origins[:max_origins]
    t = time.time()
    hv = historical_volatility.forecast(r, exp["hist_vol_window_days"], exp["trading_days_per_year"])
    gv, garch_log = garch.walk_forward_garch(r, origins, W, exp["refit_every_days"], h,
                                             exp["garch"]["p"], exp["garch"]["q"], exp["garch"]["dist"],
                                             exp["trading_days_per_year"])
    print(f"GARCH done {time.time()-t:.0f}s", flush=True)
    rf, rf_log, imp = random_forest.walk_forward_rf(data, feature_cols, origins, W, exp["refit_every_days"], h,
                                                    exp["random_forest"], exp["random_seed"])
    print(f"RF done {time.time()-t:.0f}s", flush=True)

    idx = data.index[origins]
    cfg_id = f"W{W}_h{h}_{tag}"
    long = []
    for name, s in [("historical_vol", hv), ("garch", gv), ("random_forest", rf)]:
        d = pd.DataFrame({"forecast_date": idx, "model_name": name, "predicted_volatility": s.loc[idx].values,
                          "actual_realized_volatility": data["target"].loc[idx].values})
        long.append(d)
    out = pd.concat(long, ignore_index=True)
    pos = {d: i for i, d in enumerate(data.index)}
    out["target_start_date"] = [data.index[pos[d] + 1] if pos[d] + 1 < len(data) else pd.NaT for d in out["forecast_date"]]
    out["target_end_date"] = [data.index[pos[d] + h] if pos[d] + h < len(data) else pd.NaT for d in out["forecast_date"]]
    out["asset_id"] = "EW_universe"
    out["training_window_start"] = [data.index[max(pos[d] - W + 1, 0)] for d in out["forecast_date"]]
    out["training_window_end"] = out["forecast_date"]
    out["model_config_id"] = cfg_id
    od = ROOT / "outputs" / "forecasts"
    od.mkdir(parents=True, exist_ok=True)
    out.to_csv(od / f"volatility_forecasts_{tag}.csv", index=False)
    garch_log.to_csv(od / f"garch_fit_log_{tag}.csv", index=False)
    rf_log.to_csv(od / f"rf_fit_log_{tag}.csv", index=False)
    imp.to_csv(od / f"rf_feature_importance_{tag}.csv", header=["importance"])
    data.to_parquet(ROOT / "data" / "processed" / "model_features.parquet")
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-window", type=int)
    ap.add_argument("--eval-start")
    ap.add_argument("--tag", default="main")
    ap.add_argument("--max-origins", type=int)
    a = ap.parse_args()
    run(a.train_window, a.eval_start, a.tag, a.max_origins)
