"""Parse the raw French / Cboe files into DataFrames (units untouched: percent)."""
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import ROOT

RAW = ROOT / "data" / "raw"


def _blocks_to_frames(path: Path) -> dict[str, pd.DataFrame]:
    frames = {}
    header = None
    title = "main"
    rows: dict[str, list] = {}
    heads: dict[str, list] = {}
    for line in path.read_text(errors="replace").splitlines():
        s = line.strip()
        if not s:
            continue
        parts = [c.strip() for c in s.split(",")]
        if parts[0] == "" and len(parts) > 1:
            heads[title] = parts[1:]
            rows.setdefault(title, [])
        elif parts[0].isdigit() and len(parts[0]) == 8:
            if title in heads:
                rows[title].append(parts)
        elif "--" in s:
            title = s
    for t, r in rows.items():
        df = pd.DataFrame(r, columns=["date"] + heads[t][: len(r[0]) - 1])
        df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
        frames[t] = df.set_index("date").apply(pd.to_numeric, errors="coerce")
    return frames


def load_industry(weighting: str = "value") -> pd.DataFrame:
    frames = _blocks_to_frames(RAW / "10_Industry_Portfolios_Daily.csv")
    key = "Value Weighted" if weighting == "value" else "Equal Weighted"
    for t, df in frames.items():
        if key in t:
            return df
    raise KeyError(f"no {key} block found: {list(frames)}")


def load_factors() -> pd.DataFrame:
    frames = _blocks_to_frames(RAW / "F-F_Research_Data_Factors_daily.csv")
    return next(iter(frames.values()))


def load_vix() -> pd.DataFrame:
    df = pd.read_csv(RAW / "VIX_History.csv")
    df["DATE"] = pd.to_datetime(df["DATE"], format="%m/%d/%Y")
    return df.set_index("DATE").rename_axis("date").rename(columns=str.lower)
