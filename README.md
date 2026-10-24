# Stress-Aware Machine Learning for Dynamic Portfolio Allocation

Reproducible walk-forward study: do volatility forecasts (historical, GARCH(1,1), Random Forest) improve portfolio allocation
versus equal-weight / minimum-variance / mean-variance benchmarks, after costs and across market regimes?
Data: Kenneth French daily industry portfolios + Cboe VIX, 1990–2026; out-of-sample 1993-04 → 2026-08.

**Headline (see `reports/research_report.md`):** volatility-adaptive exposure roughly halved max drawdown (−24…−28% vs −53%) and
cut volatility, at lower return; Sharpe gains (+0.03…0.06) are not statistically significant and fade with quarterly rebalancing,
high costs and in 2013–2026. The Random Forest and XGBoost beat historical volatility on QLIKE but not GARCH, and were worst in stress.

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
python scripts/download_data.py && python scripts/build_dataset.py
python scripts/run_forecasts.py --tag main
python scripts/run_xgb.py                 # optional XGBoost model (~35 min)
python scripts/run_backtest.py && python scripts/generate_report.py
pytest -q
```
Full steps (including robustness runs): `docs/reproducibility.md`.

## Layout
`configs/` parameters · `src/` data, features, models, portfolio, backtest, visualization · `scripts/` entry points ·
`notebooks/` (8 executed notebooks reading saved outputs) · `app/streamlit_app.py` (`streamlit run app/streamlit_app.py`) ·
`tests/` (23 tests: units, leakage, timing, costs, constraints) · `outputs/` tables, figures, forecasts, portfolios, metrics ·
`reports/` report, limitations, literature · `docs/` methodology, data dictionary, experiment log · `data/` manifest and instructions (raw data not committed).

## Caveats
Historical backtest on research portfolios with stylised costs; not investment advice. Data terms: see `data/README.md`.
