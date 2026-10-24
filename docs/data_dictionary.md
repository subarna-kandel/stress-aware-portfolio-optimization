# Data dictionary

All returns are **simple daily decimals** (0.01 = 1%). Kenneth French files are in percent and are divided by 100 on load
(`src/data/clean.py`); missing codes (-99.99, -999) become NaN.

| Variable | Definition | Source / file |
|---|---|---|
| `date` | Trading date (French calendar ∩ VIX calendar, 1990-01-02 …) | all |
| `asset_id` | One of 10 value-weighted industry portfolios: NoDur, Durbl, Manuf, Enrgy, HiTec, Telcm, Shops, Hlth, Utils, Other | `data/processed/market_returns.parquet` |
| `return` | Daily **total** return of the industry portfolio (not excess) | French `10_Industry_Portfolios_daily` |
| `risk_free_rate` | Daily T-bill return (`RF`) | French `F-F_Research_Data_Factors_daily` |
| `market_return` | `Mkt-RF + RF`, total US market return | French factors file |
| `vix_close` | Cboe VIX close (4 non-VIX days forward-filled, never back-filled) | Cboe `VIX_History.csv` |
| `ew_return` | Mean of the 10 industry returns each day (daily-rebalanced equal weight). The series whose volatility is forecast | derived |
| `ret_lag0..4` | `ew_return` at t, t-1, … t-4 | `data/processed/model_features.parquet` |
| `cumret_5/21/63` | Compounded EW return over the trailing window ending t | derived |
| `vol_5/21/63` | Annualised root-mean-square EW return over the trailing window ending t | derived |
| `vix_level`, `vix_chg_1/5` | VIX at t; log change over 1 / 5 days | derived |
| `drawdown_252` | EW wealth relative to its trailing-252-day peak | derived |
| `vix_vol_ratio` | log(VIX/100) − log(`vol_21`) | derived |
| `target` (`realized_volatility`) | Annualised RMS of EW returns on days t+1 … t+5, √(mean r²)·√252 | derived, label only |
| `stress_flag` | VIX at close of t-1 ≥ 25 (threshold fixed in `configs/experiment.yaml`) | derived |
| `forecast_volatility` | Annualised 5-day-ahead volatility forecast made at close of t | `outputs/forecasts/` |
| `portfolio_weight` | Risky-asset weights after rebalancing; `CASH` = 1 − Σ weights | `outputs/portfolios/portfolio_weights.csv` |
| `turnover` | Σ\|w_after − w_before(drifted)\| over risky assets on the execution day | backtest |
| `transaction_cost` | turnover × bps / 10,000, deducted from that day's return | backtest |
