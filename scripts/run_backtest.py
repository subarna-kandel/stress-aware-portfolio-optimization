"""Backtest all strategies on the saved main-run forecasts and save weights, returns, metrics."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.backtest import metrics
from src.backtest.performance import performance_table
from src.backtest.walk_forward import run_backtest
from src.config import ROOT, load_all
from src.data.align import load_processed
from src.features import market_regimes
from src.features.returns import equal_weight_return
from src.portfolio import benchmarks as B


def load_forecasts(tag: str, idx: pd.DatetimeIndex) -> pd.DataFrame:
    f = pd.read_csv(ROOT / "outputs" / "forecasts" / f"volatility_forecasts_{tag}.csv", parse_dates=["forecast_date"])
    p = f.pivot(index="forecast_date", columns="model_name", values="predicted_volatility").reindex(idx)
    xgb_path = ROOT / "outputs" / "forecasts" / "volatility_forecasts_xgb.csv"
    if tag == "main" and xgb_path.exists():
        x = pd.read_csv(xgb_path, parse_dates=["forecast_date"]).set_index("forecast_date")["predicted_volatility"]
        p["xgboost"] = x.reindex(idx)
    p["oracle"] = f[f.model_name == "garch"].set_index("forecast_date")["actual_realized_volatility"].reindex(idx)
    return p


def build_strategies(cfgs: dict, fc: pd.DataFrame, hv_fallback: pd.Series, target_vol=None, max_w=None):
    pf = cfgs["portfolio"]
    mw = max_w if max_w is not None else pf["max_asset_weight"]
    tv = target_vol if target_vol is not None else pf["vol_target"]["target_vol"]
    common = dict(cov_window=pf["cov_window_days"], max_weight=mw)
    mv = pf["mean_variance"]
    fallback_count = {}

    def adaptive(model):
        def fn(i, hist):
            v = fc[model].iloc[i]
            if not np.isfinite(v):
                v = hv_fallback.iloc[i]
                fallback_count[model] = fallback_count.get(model, 0) + 1
            return B.vol_adaptive(hist, v, tv, pf["vol_target"]["min_exposure"], pf["vol_target"]["max_exposure"])
        return fn

    strategies = {
        "EqualWeight": lambda i, h: B.equal_weight(h),
        "MinVariance": lambda i, h: B.minimum_variance(h, **common),
        "MeanVariance": lambda i, h: B.mean_variance(h, risk_aversion=mv["risk_aversion"],
                                                     shrinkage=mv["shrinkage_to_grand_mean"], **common),
        "VolTarget_HistVol": adaptive("historical_vol"),
        "VolTarget_GARCH": adaptive("garch"),
        "VolTarget_RF": adaptive("random_forest"),
        "VolTarget_Oracle*": adaptive("oracle"),   # diagnostic only: uses the realised future volatility
    }
    if "xgboost" in fc.columns:
        strategies = {**{k: v for k, v in strategies.items() if not k.endswith("*")},
                      "VolTarget_XGB": adaptive("xgboost"), "VolTarget_Oracle*": strategies["VolTarget_Oracle*"]}
    return strategies, fallback_count


def run_all(tag="main", cost_bps=None, frequency=None, target_vol=None, max_w=None, cfgs=None):
    cfgs = cfgs or load_all()
    ind, aux = load_processed()
    fc = load_forecasts(tag, ind.index)
    first = int(fc["historical_vol"].notna().values.argmax())
    r_ew = equal_weight_return(ind)
    from src.features.volatility import rolling_rms_vol
    hv = rolling_rms_vol(r_ew, cfgs["experiment"]["hist_vol_window_days"])
    strategies, fb = build_strategies(cfgs, fc, hv, target_vol, max_w)
    bt = cfgs["backtest"]
    results = {}
    for name, fn in strategies.items():
        results[name] = run_backtest(ind, aux["risk_free_rate"], fn, first, frequency or bt["rebalance_frequency"],
                                     bt["transaction_cost_bps"] if cost_bps is None else cost_bps,
                                     bt["execution_lag_days"],
                                     max_weight=max_w if max_w is not None else cfgs["portfolio"]["max_asset_weight"])
    # trim to the common evaluation window (all strategies start the day after the first execution)
    start = max(r.returns.index[0] for r in results.values())
    for r in results.values():
        r.returns = r.returns.loc[start:].iloc[1:]
    return results, aux, fb


def main():
    cfgs = load_all()
    results, aux, fb = run_all("main", cfgs=cfgs)
    rf = aux["risk_free_rate"]
    # diagnostic: static exposure equal to the average exposure of VolTarget_RF (ex-post matched, NOT tradable)
    avg_exp = results["VolTarget_RF"].returns["exposure"].mean()
    ew = results["EqualWeight"].returns
    st = pd.DataFrame(index=ew.index)
    st["gross_return"] = avg_exp * ew["gross_return"] + (1 - avg_exp) * rf.reindex(ew.index)
    st["turnover"] = avg_exp * ew["turnover"]
    st["transaction_cost"] = avg_exp * ew["transaction_cost"]
    st["net_return"] = st["gross_return"] - st["transaction_cost"]
    st["exposure"] = avg_exp
    from src.backtest.walk_forward import BacktestResult
    results["EW_StaticMatched*"] = BacktestResult(st, pd.DataFrame())

    outp, outm = ROOT / "outputs" / "portfolios", ROOT / "outputs" / "metrics"
    flag = market_regimes.stress_flag(aux["vix_close"], cfgs["experiment"]["stress"]["vix_threshold"])
    rets = []
    for name, res in results.items():
        d = res.returns.reset_index().assign(strategy_name=name)
        d["market_regime"] = np.where(flag.reindex(res.returns.index).values, "stress", "calm")
        rets.append(d)
        if len(res.weights):
            res.weights.assign(strategy_name=name).to_csv(outp / f"weights_{name.strip('*')}.csv", index=False)
    pd.concat(rets).to_csv(outp / "portfolio_returns.csv", index=False)
    w_all = pd.concat([r.weights.assign(strategy_name=n) for n, r in results.items() if len(r.weights)])
    w_all.to_csv(outp / "portfolio_weights.csv", index=False)
    perf = performance_table(results, rf)
    perf.to_csv(outm / "strategy_performance.csv", index=False)
    reg = pd.concat({n: metrics.regime_summary(r.returns["net_return"], rf, flag) for n, r in results.items()})
    reg.to_csv(outm / "regime_performance.csv")
    print(perf.set_index("strategy_name")[["ann_return", "ann_vol", "sharpe", "max_drawdown", "calmar",
                                           "ann_turnover", "avg_exposure"]].round(3))
    print("forecast fallbacks:", fb)


if __name__ == "__main__":
    main()
