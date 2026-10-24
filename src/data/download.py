"""Download raw files and record them in the data manifest."""
import csv
import hashlib
import io
import zipfile
from datetime import date
from pathlib import Path

import requests

from src.config import ROOT, load_config

RAW = ROOT / "data" / "raw"
MANIFEST = ROOT / "data" / "data_manifest.csv"
FIELDS = ["file", "source", "url", "retrieved", "frequency", "units", "notes", "sha256"]


def _get(url: str) -> bytes:
    resp = requests.get(url, timeout=120, headers={"User-Agent": "pfp-research/0.1"})
    resp.raise_for_status()
    return resp.content


def _write_manifest(rows: list[dict]) -> None:
    with open(MANIFEST, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


def download_all() -> None:
    cfg = load_config("data")
    RAW.mkdir(parents=True, exist_ok=True)
    today = date.today().isoformat()
    rows = []
    for key, fname in cfg["french_files"].items():
        url = f"{cfg['french_base_url']}/{fname}"
        content = _get(url)
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            for member in zf.namelist():
                data = zf.read(member)
                (RAW / member).write_bytes(data)
                rows.append({
                    "file": f"data/raw/{member}", "source": "Kenneth R. French Data Library",
                    "url": url, "retrieved": today, "frequency": "daily",
                    "units": "percent (not decimal); -99.99/-999 = missing",
                    "notes": ("Industry returns are total returns, not excess returns. "
                              if key == "industry_daily" else
                              "Mkt-RF, SMB, HML are excess returns; RF is the daily T-bill rate. ")
                             + "Attribution: Fama & French; check provider terms before redistribution.",
                    "sha256": hashlib.sha256(data).hexdigest(),
                })
    data = _get(cfg["vix_url"])
    (RAW / "VIX_History.csv").write_bytes(data)
    rows.append({
        "file": "data/raw/VIX_History.csv", "source": "Cboe VIX historical data",
        "url": cfg["vix_url"], "retrieved": today, "frequency": "daily",
        "units": "index points; columns OPEN,HIGH,LOW,CLOSE; dates MM/DD/YYYY",
        "notes": "Used only as a lagged market-stress feature/regime label. Check Cboe terms before redistribution.",
        "sha256": hashlib.sha256(data).hexdigest(),
    })
    _write_manifest(rows)


if __name__ == "__main__":
    download_all()
