# Data

Raw files are **not committed** (`data/raw/` is git-ignored). Re-create them with:

```bash
python scripts/download_data.py   # downloads and writes data/data_manifest.csv (source, URL, date, SHA-256)
python scripts/build_dataset.py   # cleans/aligns -> data/processed/*.parquet + data_quality_report.json
```

Sources: Kenneth R. French Data Library (Fama & French; CRSP-based), Cboe VIX historical data.
Check each provider's terms of use before redistributing raw or processed files; cite the French library and Cboe when publishing results.
Retrieval date, units and checksums are recorded in `data_manifest.csv`.
