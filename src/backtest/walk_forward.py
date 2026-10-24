"""Event loop: weights drift with returns between rebalances; trades and costs occur on execution dates."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.backtest import execution
from src.backtest.transaction_costs import cost as tc_cost
from src.backtest.transaction_costs import turnover as tc_turnover
from src.portfolio.constraints import check_weights


@dataclass
class BacktestResult:
    returns: pd.DataFrame      # date, gross_return, turnover, transaction_cost, net_return, exposure
    weights: pd.DataFrame      # rebalance_date (signal), execution_date, asset weights + cash


def run_backtest(asset_returns: pd.DataFrame, rf: pd.Series, weight_fn, first_signal_pos: int,
                 frequency: str, cost_bps: float, lag: int = 1, max_weight: float = 1.0,
                 start_pos: int | None = None) -> BacktestResult:
    """weight_fn(i, history) -> risky target weights (cash = 1 - sum). history = asset_returns.iloc[:i+1]."""
    R = asset_returns.values
    rfv = rf.reindex(asset_returns.index).values
    idx = asset_returns.index
    signals = execution.signal_positions(idx, frequency, first_signal_pos)
    exec_of = {s + lag: s for s in signals if s + lag < len(idx)}

    w = np.zeros(R.shape[1])             # risky weights held into the day, drifted; cash is the residual
    start = min(exec_of) if start_pos is None else start_pos
    rows, wlog = [], []
    for t in range(start, len(idx)):
        # 1) the day's return is earned by the weights held from the previous close
        gross = float(w @ R[t] + (1.0 - w.sum()) * rfv[t])
        w = w * (1 + R[t]) / (1 + gross)  # drift to end-of-day weights
        # 2) if t is an execution date, trade at the close of t using a signal computed at close of t-lag
        turn = c = 0.0
        if t in exec_of:
            s = exec_of[t]
            target = np.asarray(weight_fn(s, asset_returns.iloc[: s + 1]), dtype=float)
            check_weights(target, max_weight=max_weight)
            turn = tc_turnover(w, target)
            c = tc_cost(turn, cost_bps)
            w = target
            wlog.append({"rebalance_date": idx[s], "execution_date": idx[t],
                         **dict(zip(asset_returns.columns, w)), "CASH": 1 - w.sum()})
        rows.append({"date": idx[t], "gross_return": gross, "turnover": turn, "transaction_cost": c,
                     "net_return": gross - c, "exposure": w.sum()})
    return BacktestResult(pd.DataFrame(rows).set_index("date"), pd.DataFrame(wlog))
