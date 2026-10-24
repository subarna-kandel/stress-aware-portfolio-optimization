"""Read-only dashboard over saved outputs.  Run: streamlit run app/streamlit_app.py"""
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

OUT = Path(__file__).resolve().parents[1] / "outputs"
st.set_page_config(page_title="Stress-aware portfolio allocation", layout="wide")
st.title("Stress-aware portfolio allocation: saved walk-forward results")
st.caption("Historical research output, not investment advice. Regenerate with the scripts in /scripts.")


@st.cache_data
def read(path, **kw):
    return pd.read_csv(path, **kw)


tab1, tab2, tab3, tab4 = st.tabs(["Strategies", "Volatility forecasts", "Regimes", "Robustness"])

with tab1:
    perf = read(OUT / "metrics" / "strategy_performance.csv")
    st.dataframe(perf[["strategy_name", "ann_return", "ann_vol", "sharpe", "max_drawdown", "calmar",
                       "ann_turnover", "avg_exposure"]].round(3), width="stretch")
    rets = read(OUT / "portfolios" / "portfolio_returns.csv", parse_dates=["date"])
    chosen = st.multiselect("Strategies", sorted(rets.strategy_name.unique()),
                            default=["EqualWeight", "MinVariance", "VolTarget_GARCH", "VolTarget_RF"])
    d = rets[rets.strategy_name.isin(chosen)].copy()
    d["wealth"] = d.groupby("strategy_name")["net_return"].transform(lambda x: (1 + x).cumprod())
    d["drawdown"] = d["wealth"] / d.groupby("strategy_name")["wealth"].cummax() - 1
    st.plotly_chart(px.line(d, x="date", y="wealth", color="strategy_name", log_y=True,
                            title="Growth of 1, net of costs"), width="stretch")
    st.plotly_chart(px.line(d, x="date", y="drawdown", color="strategy_name", title="Drawdown"),
                    width="stretch")

with tab2:
    f = read(OUT / "forecasts" / "volatility_forecasts_main.csv", parse_dates=["forecast_date"])
    xp = OUT / "forecasts" / "volatility_forecasts_xgb.csv"
    if xp.exists():
        f = pd.concat([f, read(xp, parse_dates=["forecast_date"])])
    models = st.multiselect("Models", sorted(f.model_name.unique()), default=["garch", "random_forest"])
    lo, hi = f.forecast_date.min().date(), f.forecast_date.max().date()
    a, b = st.slider("Date range", lo, hi, (lo, hi))
    g = f[f.model_name.isin(models) & f.forecast_date.between(pd.Timestamp(a), pd.Timestamp(b))]
    act = f[f.model_name == f.model_name.iloc[0]][["forecast_date", "actual_realized_volatility"]].assign(model_name="realized")
    plot = pd.concat([g[["forecast_date", "predicted_volatility", "model_name"]].rename(columns={"predicted_volatility": "vol"}),
                      act.rename(columns={"actual_realized_volatility": "vol"})
                      .query("@pd.Timestamp(@a) <= forecast_date <= @pd.Timestamp(@b)")])
    st.plotly_chart(px.line(plot, x="forecast_date", y="vol", color="model_name",
                            title="5-day-ahead annualised volatility"), width="stretch")
    st.subheader("Accuracy (full sample)")
    st.dataframe(read(OUT / "tables" / "forecast_accuracy.csv", index_col=0).round(4))
    st.subheader("Diebold-Mariano tests")
    st.dataframe(read(OUT / "tables" / "diebold_mariano.csv").round(4))

with tab3:
    reg = read(OUT / "metrics" / "regime_performance.csv")
    reg.columns = ["strategy_name", "regime"] + list(reg.columns[2:])
    st.dataframe(reg.round(3), width="stretch")
    st.dataframe(read(OUT / "tables" / "forecast_accuracy_by_regime.csv").round(4))

with tab4:
    names = {"cost": "Transaction cost (bps)", "rebalance": "Rebalance frequency", "target_vol": "Target volatility",
             "max_weight": "Max weight", "training_window": "Training window", "subperiods": "Sub-periods",
             "stress_threshold": "Stress threshold"}
    k = st.selectbox("Sensitivity", list(names), format_func=names.get)
    st.dataframe(read(OUT / "tables" / f"robustness_{k}.csv").round(3), width="stretch")
