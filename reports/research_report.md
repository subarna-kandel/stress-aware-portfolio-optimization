# Can Machine Learning Improve Portfolio Allocation During Market Stress? A Walk-Forward Comparison of Volatility Forecasting Models and Risk-Based Strategies

*All numbers below come from the run saved in `outputs/` (main configuration in `configs/`). Nothing here is investment advice.*

## Abstract
We compare four 5-day-ahead volatility forecasts (21-day historical volatility, GARCH(1,1), Random Forest, XGBoost) for an equal-weighted basket of ten US industry portfolios, and test whether using them to scale equity exposure improves a simple portfolio relative to equal-weight, minimum-variance and constrained mean-variance benchmarks. The evaluation is strictly walk-forward from 1993-04 to 2026-08 (8,405 daily forecasts; 8,389 backtest days) with a one-day execution lag, drifting weights and proportional costs. GARCH had the lowest RMSE and QLIKE; the Random Forest beat historical volatility on QLIKE (Diebold–Mariano p = 0.002) but not on RMSE, and was not distinguishable from GARCH. Volatility-adaptive strategies roughly halved the maximum drawdown (−24% to −28% vs −53% for equal weight) and reduced volatility, at the price of lower return (8.7–8.8% vs 11.3% per year). Net Sharpe ratios were modestly higher (0.60–0.62 vs 0.56), but no Sharpe difference is statistically significant. Neither tree model translated into better portfolios than GARCH or historical volatility.

## 1. Research questions
Can volatility-adaptive allocation improve risk-adjusted performance and reduce drawdowns versus traditional strategies after costs and across regimes? Secondary: (i) does ML improve on historical volatility/GARCH; (ii) do better forecasts give better portfolios; (iii) is the benefit concentrated in stress; (iv) robustness to costs, rebalancing, windows, sub-periods.

## 2. Data
Kenneth R. French Data Library: ten value-weighted industry portfolios (daily total returns) and the daily risk-free rate; Cboe VIX. Aligned sample: see `data/processed/data_quality_report.json`: 9,233 trading days 1990-01-02 → 2026-08-31, no duplicates, no missing industry returns; 4 VIX holidays forward-filled. Returns converted from percent to decimals. Pairwise industry correlations and volatility clustering are shown in `fig_correlation_heatmap` and `fig_volatility_clustering`; VIX and subsequent 5-day realized volatility have correlation 0.74.

## 3. Methods
See `docs/methodology.md`. Target: annualised RMS of the next five daily returns of the equal-weighted basket. Rolling 756-day training window, models refit every 21 days, daily forecasts. Random Forest: 17 lagged features (returns, trailing vols, VIX, drawdown), labels embargoed so no training label overlaps the test period, hyper-parameters tuned yearly with a time-series CV inside the training window. Portfolios: EqualWeight, MinVariance, MeanVariance, and VolTarget_{HistVol,GARCH,RF} = equal-weight sleeve × min(1, 10%/forecast) with the rest in T-bills (no leverage). Monthly rebalancing, signal at close *t*, trade at close *t+1*, 5 bps cost on turnover. Diagnostics (marked `*`): an oracle that uses realized future volatility and a static-exposure benchmark matched to the average exposure of VolTarget_RF; neither is tradable. Parameters of the main run were taken from the project specification and not tuned on results.

## 4. Forecast results (all 8,405 out-of-sample days)
| Model | RMSE | MAE | QLIKE | Bias |
|---|---|---|---|---|
| Historical vol (21d) | 0.0775 | 0.0504 | 0.463 | +0.007 |
| GARCH(1,1) | **0.0713** | 0.0480 | **0.390** | +0.012 |
| Random Forest | 0.0770 | **0.0444** | 0.399 | −0.000 |
| XGBoost | 0.0809 | 0.0479 | 0.405 | +0.006 |

Diebold–Mariano (HAC, h=5): GARCH beats historical vol on QLIKE and squared error (p < 0.001, p = 0.0003). Random Forest beats historical vol on QLIKE (p = 0.002) but not squared error (p = 0.85). Random Forest vs GARCH: no significant difference (QLIKE p = 0.62; squared error p = 0.11 in GARCH's favour). XGBoost (optional Model D, same protocol and features) beats historical volatility on QLIKE (p = 0.0015) but not squared error (p = 0.24); it is not distinguishable from GARCH on QLIKE (p = 0.33) and is worse than GARCH (p = 0.003) and the Random Forest (p = 0.009) on squared error. Convergence: all 401 GARCH fits converged.

**By regime** (VIX≥25 on previous day; 1,510 stress days): QLIKE in stress — historical 0.379, GARCH 0.339, Random Forest **0.563**, XGBoost 0.499; in calm — 0.482, 0.401, **0.364**, XGBoost 0.385. The Random Forest is the best model in calm markets and the worst in stress, consistent with trees not extrapolating to volatility levels beyond those in a three-year window. By sub-period, GARCH had the lowest QLIKE in 2003–2012 and 2013–2026, the Random Forest in 1993–2002 (`forecast_accuracy_by_period.csv`). The Random Forest's most important features were VIX level (0.36) and the 252-day drawdown (0.18).

## 5. Portfolio results (net of 5 bps, 1993-04 → 2026-08)
| Strategy | Ann. return | Ann. vol | Sharpe | Max DD | Calmar | Turnover/yr | Avg. exposure |
|---|---|---|---|---|---|---|---|
| EqualWeight | 11.3% | 17.2% | 0.565 | −52.7% | 0.21 | 0.32 | 1.00 |
| MinVariance | 9.9% | 14.3% | 0.562 | −38.7% | 0.26 | 0.64 | 1.00 |
| MeanVariance | 9.4% | 14.5% | 0.526 | −39.2% | 0.24 | 1.30 | 1.00 |
| VolTarget_HistVol | 8.8% | 10.7% | 0.613 | −24.1% | 0.37 | 1.72 | 0.77 |
| VolTarget_GARCH | 8.7% | 10.4% | 0.622 | −25.3% | 0.34 | 1.37 | 0.75 |
| VolTarget_RF | 8.7% | 10.9% | 0.599 | −28.5% | 0.31 | 1.24 | 0.78 |
| VolTarget_XGB | 8.5% | 10.6% | 0.586 | −28.1% | 0.30 | 1.57 | 0.76 |
| *VolTarget_Oracle\** | 9.3% | 10.1% | 0.686 | −29.5% | 0.32 | 2.35 | 0.77 |
| *EW_StaticMatched\** | 9.6% | 13.5% | 0.565 | −43.1% | 0.22 | 0.25 | 0.78 |

Stationary-bootstrap Sharpe differences vs EqualWeight: HistVol +0.049 (p = 0.51), GARCH +0.058 (p = 0.41), RF +0.035 (p = 0.56), XGB +0.021 (p = 0.73); MinVariance −0.003 (p = 0.97). RF − GARCH = −0.023 (p = 0.52); XGB − GARCH = −0.036 (p = 0.40). **No Sharpe difference is statistically significant.** The lower drawdown, by contrast, is large and mechanical: the strategies de-risk after volatility rises. Compared with the static exposure matched to the same average exposure, VolTarget_RF cut the maximum drawdown from −43% to −28%, i.e. the timing component did reduce tail losses beyond simply holding less equity, but the Sharpe gain over the static mix is the same insignificant +0.035. Perfect knowledge of the next five days' volatility (oracle) would have lifted Sharpe to 0.69, so the realistic forecasts capture some, not all, of the available value; the oracle is not attainable and is shown only as a bound.

**Regimes** (Sharpe, stress / calm): EqualWeight 0.68 / 0.59; VolTarget_GARCH 0.78 / 0.58; VolTarget_RF 0.68 / 0.59; VolTarget_HistVol 0.67 / 0.60. Annualised volatility on stress days fell from 31% (EqualWeight) to 13–15%. Worst single stress day: −11.4% (EqualWeight) vs about −4.4% (adaptive). The benefit is therefore concentrated in stress through risk reduction, not through higher return per unit of risk. In calm periods the adaptive strategies were not better.

**Costs.** Cost drag was ≤ 0.09% per year at 5 bps. Net Sharpe at 0/5/10/25/50 bps (XGB 0.593/0.586/0.578/0.556/0.520): GARCH 0.629/0.622/0.615/0.596/0.563; HistVol 0.621/0.613/0.605/0.581/0.541; RF 0.605/0.599/0.594/0.577/0.548; EqualWeight 0.566/0.565/0.564/0.561/0.556. At 50 bps the adaptive Sharpe advantage essentially disappears.

## 6. Robustness
(Tables: `outputs/tables/robustness_*.csv`; plan in `docs/experiment_log.md`.)
* **Rebalance frequency** (Sharpe, monthly / weekly / quarterly): GARCH 0.622 / 0.583 / 0.508; RF 0.599 / 0.562 / 0.509; XGB 0.586 / 0.553 / 0.455; HistVol 0.613 / 0.571 / 0.511; EqualWeight 0.565 / 0.565 / 0.568. At quarterly rebalancing the volatility-adaptive strategies underperform equal weight on Sharpe and their drawdown advantage shrinks (−36% vs −52%): a 5-day forecast is stale after three months. The main-specification result is not robust to slow rebalancing.
* **Target volatility** 8/10/12/15%: GARCH Sharpe 0.625/0.622/0.616/0.610; RF 0.595/0.599/0.595/0.579 — insensitive.
* **Training window** 504/756/1260 days (common start 1995-04): RF QLIKE 0.414/0.404/0.383 and Sharpe 0.584/0.591/0.589; GARCH QLIKE 0.396/0.389/0.393. With the longest window the RF QLIKE (0.383) is lower than GARCH's (0.393), so the ranking of RF vs GARCH is sensitive to this choice (differences were not tested for significance).
* **Max weight** (optimised portfolios): MinVariance Sharpe 0.570/0.562/0.546 at caps 25/35/50%.
* **Stress threshold** (VIX ≥ 20/25/30): adaptive-strategy advantage on stress days grows with the threshold (GARCH stress Sharpe 0.72/0.78/1.02; EqualWeight 0.63/0.68/0.76).
* **Sub-periods** (Sharpe, GARCH / EqualWeight): 1993–2002 0.50 / 0.42; 2003–2012 0.61 / 0.47; 2013–2026 0.72 / 0.76. The Sharpe gain is not present in the most recent period. In the 2007-10→2009-06 window all adaptive strategies had a *lower* Sharpe than equal weight (−0.83 to −0.93 vs −0.51) while their window max drawdown was roughly half (−23% to −28% vs −53%): they cut losses but also participated less in the rebound. Short crisis windows (2020, 2022) are too brief for reliable ratios.

## 7. Hypotheses
* H1 (forecast accuracy): **partly supported.** RF beat historical volatility on QLIKE and MAE, not RMSE; it did not beat GARCH, and was worst in stress.
* H2 (portfolio usefulness): **supported for drawdown/volatility; not supported for return or for statistically significant Sharpe gains.**
* H3 (regimes): **supported** — the adaptive/static difference differs by regime (large vol reduction in stress; none in calm).
* H4 (costs): at 5 bps costs are small; at 50 bps the Sharpe advantage is gone.
* H5 (robustness): results are sensitive to rebalance frequency and sub-period, and to a lesser degree to the training window.
Better forecast scores did not map cleanly to better portfolios: RF had a lower MAE than GARCH but a worse portfolio Sharpe, while GARCH, the most accurate forecast, produced the best Sharpe by a margin (0.622 vs 0.599) that is not statistically distinguishable from zero.

## 8. Limitations
See `reports/limitations.md`. Main caveats: a single historical path with few independent crises; research (not investable) portfolios and stylised costs; overlapping targets; main specification not tuned but also not pre-registered externally; and the Sharpe-ratio tests have low power.

## 9. Conclusion and next steps
Volatility-adaptive exposure gave a large, robust reduction in drawdown and volatility at the price of return, but any Sharpe improvement is small, not statistically significant, and fades at slow rebalancing, high costs and in the post-2012 sample. Neither the Random Forest nor XGBoost added portfolio value over GARCH or even a rolling window in this setup. Next: a HAR baseline, regime-aware or ensemble forecasts that fix the stress-period weakness, a shorter forecast-to-rebalance gap, a second universe (ETFs / Europe), and a literature review.
