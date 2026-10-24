"""Optional Model D: walk-forward XGBoost forecasts on the same origins as the main run.

Writes outputs/forecasts/volatility_forecasts_xgb.csv (same columns as the main file, model_name = xgboost).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.config import ROOT, load_config
from src.data.align import load_processed
from src.features import build_features as BF
from src.features.returns import equal_weight_return
from src.models.xgboost_model import walk_forward_xgb


def main():
    exp = load_config("experiment")
    W, h = exp["training_window_days"], exp["forecast_horizon_days"]
    ind, aux = load_processed()
    r = equal_weight_return(ind)
    data = BF.build_dataset(r, aux["vix_close"], horizon=h, windows=exp["feature_windows"],
                            vix_lags=exp["vix_lags"], periods=exp["trading_days_per_year"])
    cols = [c for c in data.columns if c != "target"]
    main_fc = pd.read_csv(ROOT / "outputs" / "forecasts" / "volatility_forecasts_main.csv", parse_dates=["forecast_date"])
    idx = pd.DatetimeIndex(sorted(main_fc.forecast_date.unique()))
    origins = [int(data.index.get_loc(d)) for d in idx]
    pred, log, imp = walk_forward_xgb(data, cols, origins, W, exp["refit_every_days"], h, exp["xgboost"], exp["random_seed"])
    out = main_fc[main_fc.model_name == "random_forest"].copy()
    out["model_name"] = "xgboost"
    out["predicted_volatility"] = pred.loc[out.forecast_date].values
    out["model_config_id"] = out["model_config_id"].str.replace("main", "xgb")
    od = ROOT / "outputs" / "forecasts"
    out.to_csv(od / "volatility_forecasts_xgb.csv", index=False)
    log.to_csv(od / "xgb_fit_log.csv", index=False)
    imp.to_csv(od / "rf_feature_importance_xgb.csv", header=["importance"])
    print(out.predicted_volatility.describe())


if __name__ == "__main__":
    main()
