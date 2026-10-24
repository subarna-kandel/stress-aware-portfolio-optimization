import numpy as np
import pandas as pd

from src.visualization.style import COLORS, plt, save


def actual_vs_predicted(fc: pd.DataFrame, out):
    fig, ax = plt.subplots(figsize=(10, 4))
    a = fc["actual"].dropna()
    ax.plot(a.index, a, color="black", lw=0.7, label="Realized (next 5d)")
    for m in ["historical_vol", "garch", "random_forest"]:
        ax.plot(fc.index, fc[m], color=COLORS[m], lw=0.7, alpha=0.85, label=m)
    ax.set_ylabel("Annualised volatility")
    ax.set_title("Out-of-sample 5-day volatility forecasts vs realized (EW universe)")
    ax.legend(ncol=4, loc="upper left")
    save(fig, out)


def forecast_loss_bars(summary: pd.DataFrame, out):
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4))
    for ax, col in zip(axes, ["RMSE", "MAE", "QLIKE"]):
        ax.bar(summary.index, summary[col], color=[COLORS[m] for m in summary.index])
        ax.set_title(col)
        ax.tick_params(axis="x", rotation=30)
    fig.suptitle("Forecast losses, full out-of-sample period (lower is better)")
    save(fig, out)


def rolling_qlike(losses: pd.DataFrame, out, window=252):
    fig, ax = plt.subplots(figsize=(10, 3.6))
    for m in losses:
        ax.plot(losses.index, losses[m].rolling(window).mean(), color=COLORS[m], label=m)
    ax.set_yscale("log")
    ax.set_ylabel("QLIKE (rolling 1y mean, log scale)")
    ax.set_title("Forecast-loss stability across test windows")
    ax.legend()
    save(fig, out)
