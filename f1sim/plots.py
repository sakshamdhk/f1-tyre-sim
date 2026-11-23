"""Draw every figure from the saved results (no model training here).

    python -m f1sim.plots
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

R, OUT = "results", "plots"
# colour-blind-safe categorical slots, always in this order
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
COMP_COLOR = {"soft": BLUE, "medium": ORANGE, "hard": AQUA}
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#9a9893", "#e6e5e1"
LABEL = {"baseline": "Baseline (compound only)", "linear": "Linear (OLS)", "ridge": "Ridge",
         "rf": "Random forest", "gbm": "Gradient boosting", "mlp": "Neural net (MLP)",
         "degnet": "Physics-informed NN"}

plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold",
    "axes.titlesize": 12, "axes.titlelocation": "left", "font.size": 10, "lines.linewidth": 2,
    "legend.frameon": False,
})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(f"{OUT}/{name}.png", dpi=150, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    print(f"  plots/{name}.png")


def degradation_curves():
    c = pd.read_csv(f"{R}/curves_cv.csv")
    c["comp"] = c.Compound.str.lower()
    # pick the race with the most well-populated curves across compounds
    score = c.groupby(["RaceId", "comp"]).size().unstack().fillna(0)
    rid = score[(score > 15).sum(1) >= 2].sum(1).idxmax()
    races = pd.read_csv("data/races.csv", index_col="RaceId")
    g = c[c.RaceId == rid]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for comp, h in g.groupby("comp"):
        col = COMP_COLOR[comp]
        ax.scatter(h.tyre_age, h.deg, s=22, color=col, alpha=0.55, edgecolor="#fcfcfb", linewidth=1)
        ax.plot(h.tyre_age, h.degnet, color=col)
        ax.annotate(comp.capitalize(), (h.tyre_age.iloc[-1], h.degnet.iloc[-1]), xytext=(6, 0),
                    textcoords="offset points", va="center", color=INK2, fontsize=9)
    ax.axhline(0, color=MUTED, linewidth=1)
    ax.set(xlabel="Tyre age (laps)", ylabel="Time lost to tyre wear (s/lap)")
    ax.set_title(f"Tyre degradation, {races.loc[rid].Race}", pad=22)
    ax.text(0, 1.02, "dots: observed average per tyre age · lines: physics-informed NN (circuit never seen in training)",
            transform=ax.transAxes, fontsize=8.5, color=INK2, va="bottom")
    save(fig, "plot1_degradation")


def model_comparison():
    m = pd.read_csv(f"{R}/metrics.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    order = list(LABEL)
    for ax, (proto, title) in zip(axes, [("cv", "Unseen circuits (5-fold CV)"), ("temporal", "Next season (2023 → 2024)")]):
        g = m[m.protocol == proto].set_index("model").reindex(order).dropna(subset=["lap_mae"])
        y = np.arange(len(g))[::-1]
        ax.barh(y, g.lap_mae, height=0.55, color=[MUTED if i == "baseline" else BLUE for i in g.index])
        ax.errorbar(g.lap_mae, y, xerr=[g.lap_mae - g.lap_mae_lo, g.lap_mae_hi - g.lap_mae], fmt="none",
                    ecolor=INK2, elinewidth=1, capsize=3)
        for yi, v in zip(y, g.lap_mae):
            ax.text(g.lap_mae_hi.max() * 1.02, yi, f"{v:.3f}", va="center", fontsize=9, color=INK2)
        ax.set_yticks(y, [LABEL[i] for i in g.index])
        ax.set_xlim(g.lap_mae_lo.min() * 0.9, g.lap_mae_hi.max() * 1.12)
        ax.set(title=title, xlabel="Lap-level MAE (s), lower is better")
        ax.grid(axis="y", visible=False)
    fig.text(0.01, 0.0, "Whiskers: 95% bootstrap interval, resampling whole races.", fontsize=8.5, color=INK2)
    save(fig, "plot2_model_comparison")


def temperature_effect():
    t = pd.read_csv(f"{R}/temperature_effect.csv")
    s = pd.read_csv("data/stints.csv")
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), sharey=True)
    for ax, comp in zip(axes, ["soft", "medium", "hard"]):
        col = COMP_COLOR[comp]
        h = s[s.compound == comp]
        ax.scatter(h.track_temp, h.slope, s=10, color=col, alpha=0.18, linewidth=0)
        g = t[t.compound == comp]
        ax.fill_between(g.track_temp, g.lo, g.hi, color=col, alpha=0.2, linewidth=0)
        ax.plot(g.track_temp, g["mean"], color=col)
        ax.set(title=comp.capitalize(), xlabel="Track temperature (°C)")
        ax.set_ylim(-0.05, 0.25)
    axes[0].set_ylabel("Wear rate (s lost per lap of tyre age)")
    fig.suptitle("Does a hotter track wear tyres faster?", x=0.01, ha="left", fontweight="bold")
    fig.text(0.01, 0.0, "Dots: observed stints · line: physics-informed NN, other conditions held at median "
             "· band: 10–90% across ensemble members and folds", fontsize=8.5, color=INK2)
    save(fig, "plot3_temp_effect")


def pit_laps():
    p = pd.read_csv(f"{R}/pit_cv.csv").sort_values("actual")
    fig, ax = plt.subplots(figsize=(8.5, 0.42 * len(p) + 2.0))
    y = np.arange(len(p))
    ax.hlines(y, p.degnet_lo, p.degnet_hi, color=BLUE, alpha=0.3, linewidth=7)
    ax.scatter(p.degnet, y, color=BLUE, s=36, zorder=3, label="Physics-informed NN, tyre-optimal (bar: window)")
    if "degnet_cal" in p:
        ax.scatter(p.degnet_cal, y, facecolor="#fcfcfb", edgecolor=BLUE, linewidth=2, s=40, zorder=3,
                   label="NN + strategy calibration")
    ax.scatter(p.oracle, y, color=ORANGE, s=30, marker="D", zorder=3, label="Oracle (true wear rates)")
    ax.scatter(p.actual, y, color=INK, s=60, marker="|", linewidths=2.5, zorder=4, label="Actual (median one-stopper)")
    ax.set_yticks(y, [f"{r.Race}  ({r.pair})" for r in p.itertuples()], fontsize=8.5)
    ax.set(xlabel="Pit lap", title="Model pit lap vs what teams did")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.12), ncol=2, fontsize=8.5)
    save(fig, "plot4_pit_laps")


def learning_curve():
    lc = pd.read_csv(f"{R}/learning_curve.csv").groupby(["model", "frac"]).agg(
        n=("n_races", "mean"), mae=("mae", "mean")).reset_index()
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for m, col in [("ridge", AQUA), ("gbm", ORANGE), ("degnet", BLUE)]:
        g = lc[lc.model == m]
        ax.plot(g.n, g.mae, color=col, marker="o", markersize=7, markeredgecolor="#fcfcfb", markeredgewidth=2)
        ax.annotate(LABEL[m], (g.n.iloc[-1], g.mae.iloc[-1]), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=9, color=INK2)
    ax.set(xlabel="Races in the training set", ylabel="Test MAE (s)", title="Learning curve: does more data help?")
    ax.set_xlim(right=lc.n.max() * 1.3)
    save(fig, "plot5_learning_curve")


def nn_loss():
    h = pd.read_csv(f"{R}/nn_history.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, m in zip(axes, ["mlp", "degnet"]):
        g = h[(h.model == m) & (h.member == 0)]
        ax.plot(g.epoch, g.train, color=BLUE, label="Training races")
        ax.plot(g.epoch, g.val, color=ORANGE, label="Validation races")
        best = g.loc[g.val.idxmin()]
        ax.axvline(best.epoch, color=MUTED, linewidth=1, linestyle="--")
        ax.annotate("early stop", (best.epoch, g.val.max()), xytext=(4, 0), textcoords="offset points",
                    fontsize=8.5, color=INK2, va="top")
        ax.set(title=LABEL[m], xlabel="Epoch", ylabel="MSE (standardised)")
    axes[0].legend(fontsize=9)
    save(fig, "plot6_nn_loss")


def importance():
    imp = pd.read_csv(f"{R}/importance.csv")
    models = imp.model.unique()
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4.2))
    for ax, m in zip(np.atleast_1d(axes), models):
        g = imp[imp.model == m].sort_values("mean")
        ax.barh(g.feature, g["mean"] * 1000, height=0.55, color=BLUE)
        ax.set(title=LABEL[m], xlabel="Test MAE increase when shuffled (ms)")
        ax.grid(axis="y", visible=False)
    save(fig, "plot7_importance")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for f in [degradation_curves, model_comparison, temperature_effect, pit_laps, learning_curve, nn_loss,
              importance]:
        try:
            f()
        except FileNotFoundError as e:
            print(f"  skipped {f.__name__}: {e.filename} missing")
