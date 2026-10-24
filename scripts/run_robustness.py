"""Pre-planned sensitivity analyses (settings come from configs/backtest.yaml). Results go to outputs/tables."""
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.backtest import metrics
from src.backtest.performance import performance_table
from src.config import ROOT, load_all
from src.models import evaluate_forecasts as EF
from src.visualization import portfolio_plots as PP
from src.visualization.style import setup
from scripts.run_backtest import run_all

sys.path.insert(0, str(ROOT))
OUT = ROOT / "outputs"
KEEP = ["strategy_name", "ann_return", "ann_vol", "sharpe", "max_drawdown", "calmar", "ann_turnover", "ann_cost"]
LOG = []


def table(results, rf, **labels):
    p = performance_table({k: v for k, v in results.items()}, rf)[KEEP]
    for k, v in labels.items():
        p.insert(0, k, v)
    return p


def main():
    setup()
    cfgs = load_all()
    bt = cfgs["backtest"]
    ref = None
    out = {}

    rows = []
    for bps in bt["cost_sensitivity_bps"]:
        res, aux, _ = run_all("main", cost_bps=bps, cfgs=cfgs)
        rf = aux["risk_free_rate"]
        rows.append(table(res, rf, cost_bps=bps))
    out["cost"] = pd.concat(rows)
    LOG.append(("Transaction-cost sensitivity", f"bps in {bt['cost_sensitivity_bps']}, monthly rebalance"))

    rows = []
    for f in bt["rebalance_sensitivity"]:
        res, aux, _ = run_all("main", frequency=f, cfgs=cfgs)
        rows.append(table(res, rf, rebalance=f))
    out["rebalance"] = pd.concat(rows)
    LOG.append(("Rebalance-frequency sensitivity", f"{bt['rebalance_sensitivity']}, 5 bps"))

    rows = []
    for tv in bt["target_vol_sensitivity"]:
        res, aux, _ = run_all("main", target_vol=tv, cfgs=cfgs)
        rows.append(table({k: v for k, v in res.items() if k.startswith("VolTarget") or k == "EqualWeight"}, rf, target_vol=tv))
    out["target_vol"] = pd.concat(rows)
    LOG.append(("Target-volatility sensitivity", f"{bt['target_vol_sensitivity']}"))

    rows = []
    for mw in bt["max_weight_sensitivity"]:
        res, aux, _ = run_all("main", max_w=mw, cfgs=cfgs)
        rows.append(table({k: v for k, v in res.items() if k in ("MinVariance", "MeanVariance")}, rf, max_weight=mw))
    out["max_weight"] = pd.concat(rows)
    LOG.append(("Max-weight constraint sensitivity (optimised portfolios)", f"{bt['max_weight_sensitivity']}"))

    # training windows: forecasts re-run with common evaluation start
    rows, facc = [], []
    for w in bt["training_window_sensitivity"]:
        tag = f"tw{w}"
        if not (OUT / "forecasts" / f"volatility_forecasts_{tag}.csv").exists():
            print("missing forecasts for", tag)
            continue
        res, aux, _ = run_all(tag, cfgs=cfgs)
        rows.append(table(res, rf, training_window=w))
        f = pd.read_csv(OUT / "forecasts" / f"volatility_forecasts_{tag}.csv", parse_dates=["forecast_date"])
        p = f.pivot(index="forecast_date", columns="model_name", values="predicted_volatility")
        p["actual"] = f[f.model_name == "garch"].set_index("forecast_date")["actual_realized_volatility"]
        p = p.dropna()
        facc.append(EF.summarize(p["actual"], p[["historical_vol", "garch", "random_forest"]]).assign(training_window=w))
    if rows:
        out["training_window"] = pd.concat(rows)
        out["forecast_training_window"] = pd.concat(facc).rename_axis("model").reset_index()
    LOG.append(("Training-window sensitivity", f"{bt['training_window_sensitivity']} days, common start 1995-04"))

    # stress-threshold and sub-period analysis from the main run
    rets = pd.read_csv(OUT / "portfolios" / "portfolio_returns.csv", parse_dates=["date"])
    wide = rets.pivot(index="date", columns="strategy_name", values="net_return")
    vix = aux["vix_close"]
    rows = []
    for th in bt["stress_threshold_sensitivity"]:
        flag = (vix.shift(1) >= th)
        for s in wide:
            reg = metrics.regime_summary(wide[s], rf, flag)
            for regime, r in reg.iterrows():
                rows.append({"vix_threshold": th, "strategy_name": s, "regime": regime, **r.to_dict()})
    out["stress_threshold"] = pd.DataFrame(rows)
    LOG.append(("Stress-threshold sensitivity", f"VIX(t-1) >= {bt['stress_threshold_sensitivity']}"))

    rows = []
    periods = {"1993-2002": ("1993", "2002"), "2003-2012": ("2003", "2012"), "2013-2026": ("2013", "2026"),
               "GFC 2007-10 to 2009-06": ("2007-10-01", "2009-06-30"), "COVID 2020-02 to 2020-06": ("2020-02-01", "2020-06-30"),
               "2022": ("2022-01-01", "2022-12-31")}
    for lab, (a, b) in periods.items():
        for s in wide:
            x = wide[s].loc[a:b]
            rows.append({"period": lab, "strategy_name": s, "ann_return": metrics.ann_return(x),
                         "ann_vol": metrics.ann_vol(x), "sharpe": metrics.sharpe(x, rf), "max_drawdown": metrics.max_drawdown(x)})
    out["subperiods"] = pd.DataFrame(rows)
    LOG.append(("Sub-period / crisis-window analysis", ", ".join(periods)))

    for k, v in out.items():
        v.to_csv(OUT / "tables" / f"robustness_{k}.csv", index=False)
    sel = out["cost"][out["cost"].strategy_name.isin(["EqualWeight", "MinVariance", "VolTarget_GARCH", "VolTarget_RF"])]
    PP.sensitivity_lines(sel, "cost_bps", "sharpe", OUT / "figures" / "fig_cost_sensitivity",
                         "Net Sharpe ratio vs transaction cost", "cost (bps per unit turnover)")
    log = ["# Experiment log", "",
           "Sensitivity analyses were specified in `configs/backtest.yaml` before results were inspected and are reported",
           "in full (no specification was dropped). The main specification is `configs/*.yaml` as committed.", "",
           f"Generated {date.today().isoformat()} by `scripts/run_robustness.py`.", ""]
    log += [f"- **{a}**: {b}" for a, b in LOG]
    (ROOT / "docs" / "experiment_log.md").write_text("\n".join(log) + "\n")
    print("done")


if __name__ == "__main__":
    main()
