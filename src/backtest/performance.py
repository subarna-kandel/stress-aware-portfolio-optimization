"""Assemble strategy results into the standard output tables."""
import pandas as pd

from src.backtest import metrics


def performance_table(results: dict, rf: pd.Series, period_label: str = "full_oos") -> pd.DataFrame:
    rows = []
    for name, res in results.items():
        gross = metrics.summarize(res.returns, rf, "gross_return")
        net = metrics.summarize(res.returns, rf, "net_return")
        rows.append({"strategy_name": name, "evaluation_period": period_label, **net,
                     "gross_ann_return": gross["ann_return"], "gross_sharpe": gross["sharpe"]})
    return pd.DataFrame(rows)
