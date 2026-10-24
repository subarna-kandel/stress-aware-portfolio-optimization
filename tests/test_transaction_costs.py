import numpy as np

from src.backtest.transaction_costs import cost, turnover


def test_turnover_sum_abs_changes():
    assert np.isclose(turnover(np.array([0.5, 0.5]), np.array([0.25, 0.75])), 0.5)
    assert turnover(np.zeros(2), np.array([0.5, 0.5])) == 1.0   # building a fully invested book from cash


def test_cost_units_bps():
    assert np.isclose(cost(1.0, 5), 0.0005)           # 5 bps on 100% turnover
    assert np.isclose(cost(0.5, 10), 0.0005)
    assert cost(0.0, 100) == 0
