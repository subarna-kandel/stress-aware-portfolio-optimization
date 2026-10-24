"""Produce forecast-evaluation tables, statistical tests and all figures from saved outputs."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.backtest import metrics
from src.config import ROOT, load_all
from src.data.align import load_processed
from src.features.returns import equal_weight_return
from src.models import evaluate_forecasts as EF
from src.visualization import forecast_plots as FP
from src.visualization import portfolio_plots as PP
from src.visualization import risk_plots as RP
from src.visualization.style import setup

OUT = ROOT / "outputs"
MODELS = ["historical_vol", "garch", "random_forest", "xgboost"]


def load_fc(tag="main"):
    f = pd.read_csv(OUT / "forecasts" / f"volatility_forecasts_{tag}.csv", parse_dates=["forecast_date"])
    xp = OUT / "forecasts" / "volatility_forecasts_xgb.csv"
    if xp.exists() and tag == "main":
        f = pd.concat([f, pd.read_csv(xp, parse_dates=["forecast_date"])])
    p = f.pivot(index="forecast_date", columns="model_name", values="predicted_volatility")
    p = p[[m for m in MODELS if m in p.columns]]
    p["actual"] = f[f.model_name == "garch"].set_index("forecast_date")["actual_realized_volatility"]
    return p


def forecast_tables(cfgs):
    fc = load_fc()
    h = cfgs["experiment"]["forecast_horizon_days"]
    ev = fc.dropna()
    global MODELS
    MODELS = [m for m in MODELS if m in ev.columns]
    summ = EF.summarize(ev["actual"], ev[MODELS])
    summ.to_csv(OUT / "tables" / "forecast_accuracy.csv")
    losses = pd.DataFrame({m: EF.qlike(ev["actual"], ev[m]) for m in MODELS})
    se = pd.DataFrame({m: (ev[m] - ev["actual"]) ** 2 for m in MODELS})
    dm = []
    pairs = [("random_forest", "historical_vol"), ("random_forest", "garch"), ("garch", "historical_vol")]
    if "xgboost" in MODELS:
        pairs += [("xgboost", "historical_vol"), ("xgboost", "garch"), ("xgboost", "random_forest")]
    for a, b in pairs:
        for lname, L in [("QLIKE", losses), ("SqErr", se)]:
            dm.append({"model_A": a, "model_B": b, "loss": lname, **EF.diebold_mariano(L[a], L[b], h)})
    pd.DataFrame(dm).to_csv(OUT / "tables" / "diebold_mariano.csv", index=False)
    # by regime (VIX of previous day >= threshold)
    _, aux = load_processed()
    flag = (aux["vix_close"].shift(1) >= cfgs["experiment"]["stress"]["vix_threshold"]).reindex(ev.index)
    rows = []
    for name, mask in [("stress", flag), ("calm", ~flag)]:
        s = EF.summarize(ev.loc[mask, "actual"], ev.loc[mask, MODELS])
        rows.append(s.assign(regime=name))
    pd.concat(rows).rename_axis("model").to_csv(OUT / "tables" / "forecast_accuracy_by_regime.csv")
    # by calendar sub-period
    sub = []
    for a, b in [("1993", "2002"), ("2003", "2012"), ("2013", "2026")]:
        x = ev.loc[a:b]
        sub.append(EF.summarize(x["actual"], x[MODELS]).assign(period=f"{a}-{b}"))
    pd.concat(sub).rename_axis("model").to_csv(OUT / "tables" / "forecast_accuracy_by_period.csv")
    # figures
    FP.actual_vs_predicted(fc, OUT / "figures" / "fig_actual_vs_predicted_vol")
    FP.forecast_loss_bars(summ, OUT / "figures" / "fig_forecast_loss")
    FP.rolling_qlike(losses, OUT / "figures" / "fig_rolling_qlike")
    return summ


def portfolio_tables(cfgs):
    rets = pd.read_csv(OUT / "portfolios" / "portfolio_returns.csv", parse_dates=["date"])
    wide = rets.pivot(index="date", columns="strategy_name", values="net_return")
    perf = pd.read_csv(OUT / "metrics" / "strategy_performance.csv")
    _, aux = load_processed()
    rf = aux["risk_free_rate"]
    tests = []
    vt = [c for c in ["VolTarget_HistVol", "VolTarget_GARCH", "VolTarget_RF", "VolTarget_XGB"] if c in wide]
    for s in ["MinVariance", "MeanVariance"] + vt:
        tests.append({"strategy": s, "benchmark": "EqualWeight",
                      **metrics.stationary_bootstrap_sharpe_diff(wide[s], wide["EqualWeight"], rf)})
    cmp = [("VolTarget_RF", "VolTarget_HistVol"), ("VolTarget_RF", "VolTarget_GARCH"),
           ("VolTarget_RF", "EW_StaticMatched*")]
    if "VolTarget_XGB" in wide:
        cmp += [("VolTarget_XGB", "VolTarget_GARCH"), ("VolTarget_XGB", "VolTarget_RF")]
    for a, b in cmp:
        tests.append({"strategy": a, "benchmark": b, **metrics.stationary_bootstrap_sharpe_diff(wide[a], wide[b], rf)})
    pd.DataFrame(tests).to_csv(OUT / "tables" / "sharpe_difference_bootstrap.csv", index=False)

    # figures
    main = [c for c in wide.columns if c != "EW_StaticMatched*"]
    PP.cumulative(wide[main], OUT / "figures" / "fig_cumulative_net_return")
    PP.drawdowns(wide[["EqualWeight", "MinVariance"] + vt],
                 OUT / "figures" / "fig_drawdowns")
    PP.turnover_cost(perf, OUT / "figures" / "fig_turnover_cost")
    reg = pd.read_csv(OUT / "metrics" / "regime_performance.csv", index_col=[0, 1])
    PP.regime_bars(reg, OUT / "figures" / "fig_regime_sharpe")
    w = pd.read_csv(OUT / "portfolios" / "portfolio_weights.csv", parse_dates=["rebalance_date", "execution_date"])
    for s in ["MinVariance", "MeanVariance"]:
        PP.weights_history(w[w.strategy_name == s], OUT / "figures" / f"fig_weights_{s}", f"{s} weights at rebalance")
    expo = rets.pivot(index="date", columns="strategy_name", values="exposure")[
        vt]
    RP.vol_exposure(expo, OUT / "figures" / "fig_vol_adaptive_exposure")
    # gross vs net table
    perf[["strategy_name", "gross_ann_return", "ann_return", "gross_sharpe", "sharpe", "ann_turnover", "ann_cost"]] \
        .to_csv(OUT / "tables" / "gross_vs_net.csv", index=False)
    perf.to_csv(OUT / "tables" / "summary_performance.csv", index=False)


def eda(cfgs):
    ind, aux = load_processed()
    r = equal_weight_return(ind)
    desc = ind.describe().T
    desc["skew"], desc["kurt"] = ind.skew(), ind.kurt()
    desc["ann_vol"] = ind.std() * np.sqrt(252)
    desc.to_csv(OUT / "tables" / "descriptive_statistics.csv")
    RP.correlation_heatmap(ind, OUT / "figures" / "fig_correlation_heatmap")
    RP.vol_clustering(r, aux["vix_close"], OUT / "figures" / "fig_volatility_clustering")
    # VIX vs subsequent realized vol
    from src.features.volatility import forward_realized_vol
    d = pd.DataFrame({"vix": aux["vix_close"] / 100, "fwd": forward_realized_vol(r, 5)}).dropna()
    pd.DataFrame({"corr_vix_fwd_rv": [d.corr().iloc[0, 1]], "corr_log": [np.log(d).corr().iloc[0, 1]]}) \
        .to_csv(OUT / "tables" / "vix_vs_realized_vol.csv", index=False)


if __name__ == "__main__":
    setup()
    cfgs = load_all()
    (OUT / "tables").mkdir(exist_ok=True)
    (OUT / "figures").mkdir(exist_ok=True)
    eda(cfgs)
    print(forecast_tables(cfgs).round(4))
    portfolio_tables(cfgs)
    print("done")
