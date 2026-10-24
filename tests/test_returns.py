import numpy as np
import pandas as pd

from src.data import clean
from src.data.validate import quality_report
from src.features import returns as R


def test_percent_to_decimal():
    df = pd.DataFrame({"a": [1.0, -2.5]})
    assert np.allclose(clean.percent_to_decimal(df)["a"], [0.01, -0.025])


def test_missing_codes_become_nan():
    df = pd.DataFrame({"a": [1.0, -99.99, -999.0]})
    assert clean.mark_missing(df)["a"].isna().tolist() == [False, True, True]


def test_duplicate_dates_removed_and_sorted():
    idx = pd.to_datetime(["2020-01-03", "2020-01-02", "2020-01-02"])
    out = clean.drop_duplicate_dates(pd.DataFrame({"a": [1, 2, 3]}, index=idx))
    assert out.index.is_unique and out.index.is_monotonic_increasing
    assert quality_report(out, "x")["duplicate_dates"] == 0


def test_drawdown_and_cumulative():
    r = pd.Series([0.1, -0.1, 0.0])
    assert np.isclose(R.drawdown(r).min(), -0.1)
    assert np.isclose(R.cumulative_return(r, 2).iloc[1], 1.1 * 0.9 - 1)
