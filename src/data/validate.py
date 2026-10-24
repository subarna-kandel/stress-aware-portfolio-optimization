"""Data-quality checks. Returns a dict that is saved as the data-quality report."""
import pandas as pd


def quality_report(df: pd.DataFrame, name: str, max_abs_return: float | None = None) -> dict:
    """max_abs_return is in the units of df (percent for raw French data); None skips the check."""
    gaps = df.index.to_series().diff().dt.days
    return {
        "dataset": name,
        "start": str(df.index.min().date()),
        "end": str(df.index.max().date()),
        "n_rows": int(len(df)),
        "duplicate_dates": int(df.index.duplicated().sum()),
        "monotonic_index": bool(df.index.is_monotonic_increasing),
        "missing_cells": int(df.isna().sum().sum()),
        "max_calendar_gap_days": int(gaps.max()) if len(df) > 1 else 0,
        "n_abs_return_gt_threshold": (None if max_abs_return is None
                                       else int((df.abs() > max_abs_return).sum().sum())),
    }


def assert_valid(df: pd.DataFrame) -> None:
    if df.index.has_duplicates:
        raise ValueError("duplicate dates")
    if not df.index.is_monotonic_increasing:
        raise ValueError("index not sorted")
