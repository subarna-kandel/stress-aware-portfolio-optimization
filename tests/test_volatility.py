import numpy as np
import pandas as pd

from src.features.volatility import rolling_rms_vol
from src.models import historical_volatility
from src.models.evaluate_forecasts import diebold_mariano, qlike
from src.models.random_forest import training_rows


def test_rolling_vol_is_backward_looking():
    r = pd.Series(np.random.default_rng(1).normal(0, 0.01, 100))
    v = rolling_rms_vol(r, 21)
    assert v.iloc[:20].isna().all()
    expected = np.sqrt((r.iloc[0:21] ** 2).mean()) * np.sqrt(252)
    assert np.isclose(v.iloc[20], expected)
    assert np.isclose(historical_volatility.forecast(r, 21).iloc[50], v.iloc[50])


def test_qlike_zero_when_perfect_and_positive_otherwise():
    a = pd.Series([0.1, 0.2])
    assert np.allclose(qlike(a, a), 0)
    assert (qlike(a, a * 1.5) > 0).all() and (qlike(a, a * 0.5) > 0).all()


def test_chronological_training_rows_exclude_unobserved_labels():
    origin, W, h = 1000, 756, 5
    rows = training_rows(origin, W, h)
    assert rows.max() + h <= origin            # label window ends no later than the forecast origin
    assert len(rows) == W and (np.diff(rows) == 1).all()


def test_diebold_mariano_detects_difference():
    rng = np.random.default_rng(0)
    la = pd.Series(rng.normal(1.0, 0.1, 500))
    lb = pd.Series(rng.normal(1.5, 0.1, 500))
    out = diebold_mariano(la, lb, horizon=5)
    assert out["mean_loss_diff"] < 0 and out["p_value"] < 0.01
