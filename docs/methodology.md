# Methodology (as implemented)

**Universe & data.** Ten French value-weighted industry portfolios (daily total returns), French daily risk-free rate, Cboe VIX, 1990-01-02 → 2026-08-31 (9,233 aligned days).

**Forecast target.** At the close of day *t*, forecast the annualised root-mean-square return over *t+1…t+5* of the daily-rebalanced equal-weighted universe return. The zero-mean RMS (not a demeaned std) is the usual realized-volatility definition and is directly comparable with GARCH variance forecasts.

**Models** (all produce one forecast per trading day, walk-forward, rolling 756-day window, re-estimated every 21 days):
A. rolling 21-day RMS volatility;
B. GARCH(1,1) with constant mean, normal innovations; parameters estimated on the window, conditional variance filtered daily with those fixed parameters, 5-day average variance forecast; convergence flags logged (`outputs/forecasts/garch_fit_log_main.csv`);
C. Random Forest on log target (17 features: lagged returns, trailing returns/vols over 5/21/63 days, VIX level/changes, 252-day drawdown, VIX/vol ratio). Training rows are limited to those whose 5-day label is fully observed at the forecast origin (no label overlap with the test period). Hyper-parameters (depth, min leaf) chosen once a year by `TimeSeriesSplit(gap=5)` **inside the training window only**; predictions are back-transformed with a smearing factor from out-of-bag residuals.

**Forecast evaluation.** RMSE, MAE, bias on volatility; QLIKE on variance. Diebold–Mariano tests with Newey–West bandwidth h−1 and HLN correction. Targets of adjacent days overlap, so effective sample size is far smaller than *n*.

**Strategies.** 10 risky assets + cash (RF). EqualWeight; MinVariance (Ledoit–Wolf covariance, 756 days, long-only, ≤35%); MeanVariance (expected returns = 756-day mean shrunk 90% to the cross-sectional mean, λ=5, same constraints); VolTarget_{HistVol,GARCH,RF}: equal-weighted sleeve × min(1, 10% / forecast), remainder in cash — no leverage. Diagnostics (marked `*`, not tradable): VolTarget_Oracle (uses the realized future volatility; upper bound for forecast value) and EW_StaticMatched (constant exposure equal to the ex-post average exposure of VolTarget_RF, to separate "less risk" from "timing").

**Timing.** Signals use data through the close of the last trading day of the month; trades occur at the close of the next trading day; new weights earn returns from the following day. Between rebalances weights drift. Costs = 5 bps × one-way turnover on risky assets, deducted on the execution day.

**Regimes.** Stress = VIX at previous close ≥ 25 (≈18% of days).

**Inference.** Stationary bootstrap (mean block 20 days, 2,000 draws) for Sharpe-ratio differences.
