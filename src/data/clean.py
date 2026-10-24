"""Cleaning: missing-value codes, unit conversion."""
import numpy as np
import pandas as pd

MISSING_CODES = (-99.99, -999.0, -999)


def mark_missing(df: pd.DataFrame) -> pd.DataFrame:
    return df.mask(df.isin(MISSING_CODES))


def percent_to_decimal(df: pd.DataFrame) -> pd.DataFrame:
    """French-library returns are in percent; the project convention is decimals (0.01 = 1%)."""
    return df / 100.0


def drop_duplicate_dates(df: pd.DataFrame) -> pd.DataFrame:
    return df[~df.index.duplicated(keep="first")].sort_index()
