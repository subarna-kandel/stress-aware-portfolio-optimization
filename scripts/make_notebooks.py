"""Generate and execute the analysis notebooks. They read saved outputs and call src/ modules (no duplicated logic)."""
import sys
from pathlib import Path

import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks"

SETUP = '''import sys, pathlib
ROOT = pathlib.Path.cwd().resolve().parent if pathlib.Path.cwd().name == "notebooks" else pathlib.Path.cwd()
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from IPython.display import Image, display
pd.set_option("display.width", 160)
OUT = ROOT / "outputs"
def show(name): display(Image(filename=str(OUT / "figures" / f"{name}.png")))
'''

NOTEBOOKS = {
"01_data_acquisition_and_validation": [
("md", "# 01 Data acquisition and validation\nDownloading is done by `scripts/download_data.py` (not re-run here). This notebook inspects the manifest, the quality report and the processed data."),
("code", '''import json
from src.data.align import load_processed
print(pd.read_csv(ROOT / "data" / "data_manifest.csv")[["file", "source", "retrieved", "frequency", "units"]].to_string())'''),
("code", '''print(pd.DataFrame(json.loads((ROOT / "data/processed/data_quality_report.json").read_text())).T.to_string())'''),
("code", '''ind, aux = load_processed()
print(ind.shape, ind.index.min().date(), ind.index.max().date())
print("duplicate dates:", ind.index.duplicated().sum(), "| missing cells:", int(ind.isna().sum().sum()))
print("largest |daily return|:", float(ind.abs().max().max()))   # decimals: ~0.2 = 20%
aux.describe().T'''),
("code", '''gaps = ind.index.to_series().diff().dt.days
print(gaps.value_counts().sort_index().head(10))'''),
],
"02_exploratory_data_analysis": [
("md", "# 02 Exploratory data analysis"),
("code", "pd.read_csv(OUT / 'tables' / 'descriptive_statistics.csv', index_col=0).round(3)"),
("code", "show('fig_correlation_heatmap')"),
("code", "show('fig_volatility_clustering')"),
("code", '''from src.data.align import load_processed
from src.features.returns import equal_weight_return, drawdown
from src.features.volatility import forward_realized_vol
ind, aux = load_processed()
r = equal_weight_return(ind)
fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
(1 + r).cumprod().plot(ax=ax[0], logy=True, title="EW universe, growth of 1")
drawdown(r).plot(ax=ax[1], title="EW universe drawdown")
plt.show()'''),
("code", '''d = pd.DataFrame({"vix": aux["vix_close"] / 100, "fwd_rv5": forward_realized_vol(r, 5)}).dropna()
print(d.corr().round(3))
d.sample(2000, random_state=0).plot.scatter("vix", "fwd_rv5", alpha=0.3, title="VIX vs next-5-day realized vol")
plt.show()'''),
("code", '''flag = aux["vix_close"].shift(1) >= 25
print("stress share of days:", round(float(flag.mean()), 3))
pd.DataFrame({"stress": r[flag].describe(), "calm": r[~flag].describe()})'''),
],
"03_volatility_target_and_features": [
("md", "# 03 Volatility target and features\nTarget at *t* = annualised RMS of returns *t+1..t+5*. Features at *t* use data up to *t* only."),
("code", '''from src.data.align import load_processed
from src.features.build_features import build_dataset
from src.features.returns import equal_weight_return
ind, aux = load_processed()
r = equal_weight_return(ind)
data = build_dataset(r, aux["vix_close"], horizon=5)
data.dropna().tail()'''),
("code", '''# leakage check: wreck the future after t and confirm features at <= t are unchanged
t = 4000
r2, v2 = r.copy(), aux["vix_close"].copy()
r2.iloc[t + 1:] *= 5; v2.iloc[t + 1:] *= 3
alt = build_dataset(r2, v2, horizon=5)
feats = [c for c in data.columns if c != "target"]
print("features identical up to t:", data[feats].iloc[:t + 1].equals(alt[feats].iloc[:t + 1]))
print("target at t unchanged by returns <= t:", np.isclose(data.target.iloc[t], build_dataset(r.where(r.index > r.index[t], 0.123), aux["vix_close"]).target.iloc[t]))'''),
("code", '''dd = pd.DataFrame({"feature": feats})
dd["description"] = ["see docs/data_dictionary.md"] * len(feats)
print(len(feats), "features:", ", ".join(feats))'''),
("code", "data[['target']].dropna().describe().T"),
],
"04_baseline_and_garch_models": [
("md", "# 04 Baseline and GARCH models (saved walk-forward output)"),
("code", '''f = pd.read_csv(OUT / "forecasts" / "volatility_forecasts_main.csv", parse_dates=["forecast_date"])
p = f.pivot(index="forecast_date", columns="model_name", values="predicted_volatility")
p["actual"] = f[f.model_name == "garch"].set_index("forecast_date")["actual_realized_volatility"]
from src.models.evaluate_forecasts import summarize
ev = p.dropna()
summarize(ev["actual"], ev[["historical_vol", "garch"]]).round(4)'''),
("code", '''g = pd.read_csv(OUT / "forecasts" / "garch_fit_log_main.csv", parse_dates=["origin_date"])
print("fits:", len(g), "| convergence flags:", g.convergence_flag.value_counts().to_dict(), "| fallbacks:", int(g.used_previous_params.sum()))
g.set_index("origin_date")[["alpha[1]", "beta[1]"]].plot(title="GARCH parameters over time")
plt.show()'''),
("code", "show('fig_actual_vs_predicted_vol')"),
],
"05_random_forest_forecasting": [
("md", "# 05 Random Forest (and optional XGBoost) forecasting"),
("code", '''pd.read_csv(OUT / "tables" / "forecast_accuracy.csv", index_col=0).round(4)'''),
("code", "pd.read_csv(OUT / 'tables' / 'diebold_mariano.csv').round(4)"),
("code", "pd.read_csv(OUT / 'tables' / 'forecast_accuracy_by_regime.csv').round(4)"),
("code", '''imp = pd.read_csv(OUT / "forecasts" / "rf_feature_importance_main.csv", index_col=0)["importance"]
imp.sort_values().plot.barh(title="Random Forest feature importance (mean over refits)")
plt.show()'''),
("code", '''log = pd.read_csv(OUT / "forecasts" / "rf_fit_log_main.csv", parse_dates=["origin_date", "train_start", "train_end"])
print("max train_end - origin gap (days):", (log.origin_date - log.train_end).dt.days.min(), "(label embargo: >= 5 trading days)")
log[["max_depth", "min_samples_leaf"]].value_counts()'''),
("code", "show('fig_forecast_loss'); show('fig_rolling_qlike')"),
],
"06_portfolio_construction": [
("md", "# 06 Portfolio construction\nWeight rules evaluated on one date using only past data."),
("code", '''from src.data.align import load_processed
from src.portfolio import benchmarks as B
from src.portfolio.constraints import check_weights
ind, aux = load_processed()
i = ind.index.get_loc(pd.Timestamp("2008-08-29"))
hist = ind.iloc[: i + 1]
w = pd.DataFrame({
    "EqualWeight": B.equal_weight(hist),
    "MinVariance": B.minimum_variance(hist, cov_window=756, max_weight=0.35),
    "MeanVariance": B.mean_variance(hist, cov_window=756, max_weight=0.35, risk_aversion=5, shrinkage=0.9),
}, index=ind.columns)
for c in w: check_weights(w[c].values, 0.35)
w.round(3)'''),
("code", '''w.plot.bar(figsize=(9, 3.5), title="Weights decided at close of 2008-08-29"); plt.show()
print("volatility-adaptive exposure for forecast vols 10%/20%/40% (target 10%):",
      [round(float(B.vol_adaptive(hist, v, 0.10).sum()), 3) for v in (0.10, 0.20, 0.40)])'''),
("code", "show('fig_weights_MinVariance'); show('fig_weights_MeanVariance')"),
],
"07_walk_forward_backtest": [
("md", "# 07 Walk-forward backtest (saved results)\nTiming: signal at close *t*, trade at close *t+1*, new weights earn from *t+2*; 5 bps cost on turnover."),
("code", '''perf = pd.read_csv(OUT / "metrics" / "strategy_performance.csv").set_index("strategy_name")
perf[["ann_return", "ann_vol", "sharpe", "sortino", "max_drawdown", "calmar", "ann_turnover", "ann_cost", "avg_exposure"]].round(3)'''),
("code", "pd.read_csv(OUT / 'tables' / 'gross_vs_net.csv').round(4)"),
("code", "show('fig_cumulative_net_return'); show('fig_drawdowns')"),
("code", "pd.read_csv(OUT / 'tables' / 'sharpe_difference_bootstrap.csv').round(3)"),
("code", '''w = pd.read_csv(OUT / "portfolios" / "portfolio_weights.csv", parse_dates=["rebalance_date", "execution_date"])
print("min lag between signal and execution (days):", (w.execution_date - w.rebalance_date).dt.days.min())
w.head()'''),
("code", "show('fig_turnover_cost'); show('fig_vol_adaptive_exposure')"),
],
"08_robustness_and_regime_analysis": [
("md", "# 08 Robustness and regime analysis\nAll specifications are listed in `docs/experiment_log.md` and reported in full."),
("code", '''def piv(name, col, sharpe="sharpe"):
    d = pd.read_csv(OUT / "tables" / f"robustness_{name}.csv")
    return d.pivot(index="strategy_name", columns=col, values=sharpe).round(3)
print("Net Sharpe by transaction cost (bps)"); display(piv("cost", "cost_bps"))
print("by rebalance frequency"); display(piv("rebalance", "rebalance"))
print("by target volatility"); display(piv("target_vol", "target_vol"))
print("by training window"); display(piv("training_window", "training_window"))
print("by max weight"); display(piv("max_weight", "max_weight"))'''),
("code", '''sp = pd.read_csv(OUT / "tables" / "robustness_subperiods.csv")
display(sp.pivot(index="strategy_name", columns="period", values="sharpe").round(2))
display(sp.pivot(index="strategy_name", columns="period", values="max_drawdown").round(2))'''),
("code", '''st_ = pd.read_csv(OUT / "tables" / "robustness_stress_threshold.csv")
st_[st_.strategy_name.isin(["EqualWeight", "VolTarget_GARCH", "VolTarget_RF"])].pivot_table(
    index=["strategy_name", "regime"], columns="vix_threshold", values="sharpe").round(2)'''),
("code", "pd.read_csv(OUT / 'tables' / 'robustness_forecast_training_window.csv').pivot(index='model', columns='training_window', values='QLIKE').round(3)"),
("code", "show('fig_regime_sharpe'); show('fig_cost_sensitivity')"),
],
}


def main(execute=True):
    NB.mkdir(exist_ok=True)
    for name, cells in NOTEBOOKS.items():
        nb = nbf.v4.new_notebook()
        nb.cells.append(nbf.v4.new_code_cell(SETUP))
        for kind, src in cells:
            nb.cells.append(nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src))
        if execute:
            ExecutePreprocessor(timeout=600, kernel_name="python3").preprocess(nb, {"metadata": {"path": str(NB)}})
        nbf.write(nb, NB / f"{name}.ipynb")
        print("wrote", name)


if __name__ == "__main__":
    main("--no-exec" not in sys.argv)
