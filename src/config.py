"""Configuration loading helpers."""
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(name: str) -> dict:
    with open(ROOT / "configs" / f"{name}.yaml") as fh:
        return yaml.safe_load(fh)


def load_all() -> dict:
    return {n: load_config(n) for n in ("data", "experiment", "portfolio", "backtest")}
