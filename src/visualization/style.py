import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {
    "EqualWeight": "#4c78a8", "MinVariance": "#72b7b2", "MeanVariance": "#b279a2",
    "VolTarget_HistVol": "#f58518", "VolTarget_GARCH": "#e45756", "VolTarget_RF": "#2ca02c",
    "VolTarget_XGB": "#9467bd", "xgboost": "#9467bd", "VolTarget_Oracle*": "#999999", "EW_StaticMatched*": "#bbbbbb",
    "historical_vol": "#f58518", "garch": "#e45756", "random_forest": "#2ca02c",
}


def setup():
    plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 160, "axes.grid": True, "grid.alpha": 0.25,
                         "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
                         "axes.titlesize": 11, "legend.frameon": False})


def save(fig, path_no_ext):
    fig.tight_layout()
    fig.savefig(f"{path_no_ext}.png")
    fig.savefig(f"{path_no_ext}.pdf")
    plt.close(fig)
