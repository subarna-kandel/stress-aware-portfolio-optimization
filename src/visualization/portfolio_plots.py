import numpy as np
import pandas as pd

from src.visualization.style import COLORS, plt, save


def cumulative(rets: pd.DataFrame, out, title="Cumulative net return (growth of 1)"):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for c in rets:
        ax.plot(rets.index, (1 + rets[c]).cumprod(), color=COLORS.get(c), label=c,
                lw=1.0 if "*" not in c else 0.8, ls="-" if "*" not in c else "--")
    ax.set_yscale("log")
    ax.set_ylabel("Growth of 1 (log scale)")
    ax.set_title(title)
    ax.legend(ncol=2)
    save(fig, out)


def drawdowns(rets: pd.DataFrame, out):
    fig, ax = plt.subplots(figsize=(10, 4))
    for c in rets:
        w = (1 + rets[c]).cumprod()
        ax.plot(rets.index, w / w.cummax() - 1, color=COLORS.get(c), label=c, lw=0.9)
    ax.set_ylabel("Drawdown")
    ax.set_title("Drawdown of net returns")
    ax.legend(ncol=2)
    save(fig, out)


def weights_history(w: pd.DataFrame, out, title):
    d = w.set_index("rebalance_date").drop(columns=["execution_date", "strategy_name"], errors="ignore")
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.stackplot(d.index, d.T.values, labels=d.columns)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Portfolio weight")
    ax.set_title(title)
    ax.legend(ncol=6, fontsize=7, loc="lower center")
    save(fig, out)


def turnover_cost(perf: pd.DataFrame, out):
    p = perf.set_index("strategy_name")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    axes[0].bar(p.index, p["ann_turnover"], color=[COLORS.get(i) for i in p.index])
    axes[0].set_title("Annualised turnover (sum |dw|)")
    axes[1].bar(p.index, p["ann_cost"] * 1e4, color=[COLORS.get(i) for i in p.index])
    axes[1].set_title("Annual transaction-cost drag (bps)")
    for ax in axes:
        ax.tick_params(axis="x", rotation=45)
        for l in ax.get_xticklabels():
            l.set_ha("right")
    save(fig, out)


def regime_bars(reg: pd.DataFrame, out):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=False)
    for ax, regime in zip(axes, ["calm", "stress"]):
        d = reg.xs(regime, level=1)
        ax.bar(d.index, d["sharpe"], color=[COLORS.get(i) for i in d.index])
        ax.set_title(f"Sharpe ratio, {regime} days")
        ax.tick_params(axis="x", rotation=45)
        for l in ax.get_xticklabels():
            l.set_ha("right")
    save(fig, out)


def sensitivity_lines(df: pd.DataFrame, x: str, y: str, out, title, xlabel):
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for s, g in df.groupby("strategy_name"):
        ax.plot(g[x].astype(str), g[y], marker="o", label=s, color=COLORS.get(s))
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(y)
    ax.legend(fontsize=7, ncol=2)
    save(fig, out)
