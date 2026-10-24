# Experiment log

Sensitivity analyses were specified in `configs/backtest.yaml` before results were inspected and are reported
in full (no specification was dropped). The main specification is `configs/*.yaml` as committed.

Generated 2026-10-24 by `scripts/run_robustness.py`.

- **Transaction-cost sensitivity**: bps in [0, 5, 10, 25, 50], monthly rebalance
- **Rebalance-frequency sensitivity**: ['weekly', 'monthly', 'quarterly'], 5 bps
- **Target-volatility sensitivity**: [0.08, 0.1, 0.12, 0.15]
- **Max-weight constraint sensitivity (optimised portfolios)**: [0.25, 0.35, 0.5]
- **Training-window sensitivity**: [504, 756, 1260] days, common start 1995-04
- **Stress-threshold sensitivity**: VIX(t-1) >= [20, 25, 30]
- **Sub-period / crisis-window analysis**: 1993-2002, 2003-2012, 2013-2026, GFC 2007-10 to 2009-06, COVID 2020-02 to 2020-06, 2022
