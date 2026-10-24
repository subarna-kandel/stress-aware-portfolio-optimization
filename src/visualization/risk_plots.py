import numpy as np
import pandas as pd

from src.visualization.style import COLORS, plt, save


def correlation_heatmap(ret: pd.DataFrame, out):
    c = ret.corr()
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    im = ax.imshow(c.values, vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(len(c)), c.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(c)), c.columns)
    for i in range(len(c)):
        for j in range(len(c)):
            ax.text(j, i, f"{c.values[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if c.values[i, j] < 0.7 else "black")
    ax.grid(False)
    ax.set_title("Daily return correlation, 10 industry portfolios")
    fig.colorbar(im, ax=ax, shrink=0.8)
    save(fig, out)


def vol_clustering(r: pd.Series, vix: pd.Series, out):
    from src.features.volatility import rolling_rms_vol
    fig, axes = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True)
    axes[0].plot(r.index, r, lw=0.4, color="#4c78a8")
    axes[0].set_ylabel("Daily return")
    axes[0].set_title("Volatility clustering: EW universe returns, rolling vol and VIX")
    axes[1].plot(r.index, rolling_rms_vol(r, 21), color="#e45756", label="21d realized vol (ann.)")
    axes[1].plot(vix.index, vix / 100, color="black", lw=0.6, label="VIX / 100")
    axes[1].legend()
    save(fig, out)


def vol_exposure(exposure: pd.DataFrame, out):
    fig, ax = plt.subplots(figsize=(10, 3.6))
    for c in exposure:
        ax.plot(exposure.index, exposure[c], color=COLORS.get(c), lw=0.8, label=c)
    ax.set_ylabel("Risky-asset exposure")
    ax.set_title("Exposure of volatility-adaptive strategies")
    ax.legend()
    save(fig, out)
