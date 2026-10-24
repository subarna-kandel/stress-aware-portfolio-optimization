"""Decision timing: weights decided at close of s are traded at close of s+lag and earn from s+lag+1."""
import numpy as np
import pandas as pd

from src.backtest.execution import signal_positions
from src.backtest.walk_forward import run_backtest


def _market(n=120, k=3, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    R = pd.DataFrame(rng.normal(0.0005, 0.01, (n, k)), index=idx, columns=list("ABC"))
    rf = pd.Series(0.0001, index=idx)
    return R, rf


def test_signal_dates_are_last_trading_day_of_month():
    idx = pd.bdate_range("2020-01-01", periods=120)
    pos = signal_positions(idx, "monthly", 0)
    for p in pos:
        assert idx[p].month != idx[p + 1].month


def test_weight_fn_never_sees_future_and_returns_after_trade():
    R, rf = _market()
    seen = []

    def fn(i, hist):
        seen.append((i, len(hist), hist.index[-1]))
        return np.array([1.0, 0.0, 0.0])

    res = run_backtest(R, rf, fn, 30, "monthly", 0.0, lag=1)
    for i, n, last in seen:
        assert n == i + 1 and last == R.index[i]       # history ends at the signal date
    wlog = res.weights
    s_date, x_date = wlog.iloc[0]["rebalance_date"], wlog.iloc[0]["execution_date"]
    assert R.index.get_loc(x_date) == R.index.get_loc(s_date) + 1
    # day of execution earns nothing from the new weights; next day earns asset A
    d1 = R.index.get_loc(x_date) + 1
    assert np.isclose(res.returns.loc[R.index[d1], "gross_return"], R["A"].iloc[d1])
    assert np.isclose(res.returns.loc[x_date, "gross_return"], rf.iloc[0])  # still all cash


def test_future_shock_does_not_change_past_returns():
    R, rf = _market()
    fn = lambda i, h: np.full(3, 1 / 3)
    base = run_backtest(R, rf, fn, 30, "monthly", 5.0)
    R2 = R.copy()
    R2.iloc[80:] *= 10
    alt = run_backtest(R2, rf, fn, 30, "monthly", 5.0)
    cut = R.index[79]                                   # shock starts at row 80
    pd.testing.assert_series_equal(base.returns["net_return"].loc[:cut], alt.returns["net_return"].loc[:cut])


def test_costs_reduce_net_return_and_turnover_charged_once():
    R, rf = _market()
    fn = lambda i, h: np.full(3, 1 / 3)
    free = run_backtest(R, rf, fn, 30, "monthly", 0.0).returns
    paid = run_backtest(R, rf, fn, 30, "monthly", 10.0).returns
    assert np.allclose(free["gross_return"], paid["gross_return"])
    assert paid["transaction_cost"].sum() > 0
    first = paid[paid.turnover > 0].iloc[0]
    assert np.isclose(first["turnover"], 1.0) and np.isclose(first["transaction_cost"], 0.001)
    assert (paid["net_return"] <= free["net_return"] + 1e-15).all()


def test_reproducible():
    R, rf = _market()
    fn = lambda i, h: np.full(3, 1 / 3)
    a = run_backtest(R, rf, fn, 30, "monthly", 5.0).returns
    b = run_backtest(R, rf, fn, 30, "monthly", 5.0).returns
    pd.testing.assert_frame_equal(a, b)
