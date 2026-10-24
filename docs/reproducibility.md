# Reproducibility

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_data.py
python scripts/build_dataset.py
python scripts/run_forecasts.py --tag main                                   # ~5 min on 4 cores
for w in 504 756 1260; do python scripts/run_forecasts.py --train-window $w --eval-start 1995-04-03 --tag tw$w; done
python scripts/run_xgb.py                                                    # optional, ~35 min
python scripts/run_backtest.py
python scripts/run_robustness.py
python scripts/generate_report.py
python scripts/make_notebooks.py                                             # executes notebooks/*.ipynb
streamlit run app/streamlit_app.py                                           # optional dashboard
pytest -q
```

* All stochastic components use `random_seed: 42` (`configs/experiment.yaml`). Tested package versions are pinned in `requirements.txt`.
* The French library is re-issued periodically; a fresh download may extend the sample or revise history, so numbers can shift slightly.
  The retrieval date and file hashes are in `data/data_manifest.csv`.
* The main-specification configuration is `configs/*.yaml`; the report states the values used.
