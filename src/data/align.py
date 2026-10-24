"""Align returns, risk-free rate and VIX on common trading dates and persist the dataset."""
import json

import pandas as pd

from src.config import ROOT, load_config
from src.data import clean, load, validate

PROCESSED = ROOT / "data" / "processed"


def build_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (market_returns long-ish wide frame, aux frame with rf and vix). All returns are decimals."""
    cfg = load_config("data")
    ind = clean.drop_duplicate_dates(clean.mark_missing(load.load_industry(cfg["return_weighting"])))
    fac = clean.drop_duplicate_dates(clean.mark_missing(load.load_factors()))
    vix = clean.drop_duplicate_dates(load.load_vix())

    rf = clean.percent_to_decimal(fac[["RF"]]).rename(columns={"RF": "risk_free_rate"})
    mkt = clean.percent_to_decimal(fac[["Mkt-RF"]]).rename(columns={"Mkt-RF": "mkt_excess"})

    ind = ind[cfg["assets"]]
    # raw units are percent, so the plausibility threshold is 30 (= 30% in one day)
    reports = [validate.quality_report(ind, "industry_daily", 30.0),
               validate.quality_report(fac, "factors_daily", 30.0),
               validate.quality_report(vix[["close"]], "vix_daily")]

    ind = clean.percent_to_decimal(ind)
    aux = rf.join(mkt, how="inner").join(vix[["close"]].rename(columns={"close": "vix_close"}), how="left")
    aux["market_return"] = aux["mkt_excess"] + aux["risk_free_rate"]
    aux = aux.drop(columns="mkt_excess")

    # common trading dates: industry returns and rf must both exist; VIX is forward-filled only
    # across non-VIX holidays that are French trading days (never backward-filled).
    idx = ind.index.intersection(aux.index)
    idx = idx[idx >= pd.Timestamp(cfg["start_date"])]
    if cfg["end_date"]:
        idx = idx[idx <= pd.Timestamp(cfg["end_date"])]
    ind, aux = ind.loc[idx], aux.loc[idx]
    n_vix_missing = int(aux["vix_close"].isna().sum())
    aux["vix_close"] = aux["vix_close"].ffill()
    ind = ind.dropna(how="any")
    aux = aux.loc[ind.index].dropna()
    validate.assert_valid(ind)
    validate.assert_valid(aux)

    reports.append({"dataset": "aligned", "start": str(ind.index.min().date()), "end": str(ind.index.max().date()),
                    "n_rows": int(len(ind)), "vix_missing_before_ffill": n_vix_missing,
                    "units": "decimal returns (0.01 = 1%)"})
    PROCESSED.mkdir(parents=True, exist_ok=True)
    long = ind.rename_axis("date").stack().rename("return").rename_axis(["date", "asset_id"]).reset_index()
    long.to_parquet(PROCESSED / "market_returns.parquet", index=False)
    aux.rename_axis("date").to_parquet(PROCESSED / "aux_series.parquet")
    (PROCESSED / "data_quality_report.json").write_text(json.dumps(reports, indent=2))
    return ind, aux


def load_processed() -> tuple[pd.DataFrame, pd.DataFrame]:
    long = pd.read_parquet(PROCESSED / "market_returns.parquet")
    ind = long.pivot(index="date", columns="asset_id", values="return")
    cfg = load_config("data")
    ind = ind[cfg["assets"]]
    aux = pd.read_parquet(PROCESSED / "aux_series.parquet")
    return ind, aux
