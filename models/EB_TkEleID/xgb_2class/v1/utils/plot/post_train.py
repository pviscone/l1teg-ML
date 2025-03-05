import numpy as np
import os

import matplotlib.pyplot as plt
import mplhep as hep
import hist
from mplhep import error_estimation
from sklearn.metrics import roc_curve, roc_auc_score
import xgboost as xgb

hep.set_style("CMS")

php_index = os.path.join(os.path.dirname(__file__), "index.php")


def poisson_interval_ignore_empty(sumw, sumw2):
    # Set to 0 yerr of empty bins
    interval = error_estimation.poisson_interval(sumw, sumw2)
    lo, hi = interval[0, ...], interval[1, ...]
    to_ignore = np.isnan(lo)
    lo[to_ignore] = 0.0
    hi[to_ignore] = 0.0
    res = np.array([lo, hi])
    return np.abs(res - sumw)


def plot_importance(model, save=None):
    xgb.plot_importance(model, importance_type="gain", show_values=False)

    if save:
        plt.savefig(f"{save}_averagegain.pdf")
        plt.savefig(f"{save}_averagegain.png")
    xgb.plot_importance(model, importance_type="weight", show_values=False)

    if save:
        plt.savefig(f"{save}_weight.pdf")
        plt.savefig(f"{save}_weight.png")

    rank = {}
    for key in model.get_score():
        rank[key] = (
            model.get_score(importance_type="weight")[key]
            * model.get_score(importance_type="gain")[key]
        )

    fig, ax = plt.subplots()
    sorted_rank = {k: v for k, v in sorted(rank.items(), key=lambda item: item[1])}

    ax.barh(list(sorted_rank.keys()), width=sorted_rank.values())
    ax.set_xlabel("Gain")

    hep.cms.text("Phase-2 Simulation Preliminary", fontsize=18, ax=ax)
    hep.cms.lumitext("PU 200 (14 TeV)", fontsize=18, ax=ax)

    if save:
        os.makedirs(os.path.dirname(save), exist_ok=True)
        os.system(f"cp -n {php_index} {os.path.dirname(save)}")
        plt.savefig(f"{save}_gain.pdf")
        plt.savefig(f"{save}_gain.png")
    return sorted_rank


def plot_loss(eval_result, loss="logloss", save=False):
    fig, ax = plt.subplots()
    ax.plot(eval_result["train"][loss], label="train")
    ax.plot(eval_result["eval"][loss], label="eval")
    # ax.set_yscale("log")
    ax.set_xlabel("Boosting Round")
    ax.set_ylabel("LogLoss")
    ax.legend()
    plt.show()

    hep.cms.text("Phase-2 Simulation Preliminary", fontsize=18, ax=ax)
    hep.cms.lumitext("PU 200 (14 TeV)", fontsize=18, ax=ax)

    if save:
        os.makedirs(os.path.dirname(save), exist_ok=True)
        os.system(f"cp -n {php_index} {os.path.dirname(save)}")
        fig.savefig(save + ".png")
        fig.savefig(save + ".pdf")
    return ax


def plot_scores(
    df_train,
    df_test=None,
    weight=None,
    score=None,
    label=None,
    bins=np.linspace(0, 1, 30),
    save=False,
    log=False,
    func=lambda x: x,
):
    fig, ax = plt.subplots()

    classes = np.unique(df_train[label])

    acab_palette = (
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    )

    for idx, cls in enumerate(list(classes)):
        color = acab_palette[idx % len(acab_palette)]

        pred_train = df_train[score][df_train[label] == cls]

        train_h = hist.Hist(hist.axis.Variable(bins), storage=hist.storage.Weight())
        train_h.fill(func(pred_train), weight=df_train[weight] if weight else None)

        train_h.plot(
            label=f"Class {int(cls)} Train",
            yerr=poisson_interval_ignore_empty(train_h.values(), train_h.variances()),
            histtype="step",
            color=color,
            linewidth=2,
            density=True,
            ax=ax,
        )

        if df_test is not None:
            pred_test = df_test[score][df_test[label] == cls]

            test_h = hist.Hist(hist.axis.Variable(bins), storage=hist.storage.Weight())
            test_h.fill(func(pred_test), weight=df_test[weight] if weight else None)

            test_h.plot(
                label=f"Class {int(cls)} Test",
                color=color,
                yerr=poisson_interval_ignore_empty(test_h.values(), test_h.variances()),
                markeredgecolor="black",
                ecolor="black",
                histtype="errorbar",
                marker="^",
                density=True,
                ax=ax,
            )

    hep.cms.text("Phase-2 Simulation Preliminary", fontsize=18, ax=ax)
    hep.cms.lumitext("PU 200 (14 TeV)", fontsize=18, ax=ax)

    ax.set_xlabel("Score")
    ax.set_ylabel("Density")
    if log:
        ax.set_yscale("log")
    ax.legend(fontsize=18)

    if save:
        os.makedirs(os.path.dirname(save), exist_ok=True)
        os.system(f"cp -n {php_index} {os.path.dirname(save)}")
        fig.savefig(save + ".png")
        fig.savefig(save + ".pdf")
    return fig, ax


def plot_roc(*dfs, legend=None, score=None, label=None, weight=None, save=None):
    fig, ax = plt.subplots()

    if legend is not None:
        assert len(legend) == len(dfs), (
            "Number of legends must match number of dataframes"
        )
    else:
        legend = [""] * len(dfs)

    for df, lab in zip(dfs, legend, strict=False):
        fpr, tpr, _ = roc_curve(
            df[label], df[score], sample_weight=df[weight] if weight else None
        )
        auc = roc_auc_score(
            df[label], df[score], sample_weight=df[weight] if weight else None
        )

        if lab:
            lab = f"{lab} (AUC = {auc:.2f})"
        elif df.attrs.get("name", False):
            lab = f"{df.attrs['name']} (AUC = {auc:.2f})"
        else:
            lab = f"AUC = {auc:.2f}"
        ax.plot(fpr, tpr, label=lab)

        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", alpha=0.2)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(fontsize=18)

        hep.cms.text("Phase-2 Simulation Preliminary", fontsize=18, ax=ax)
        hep.cms.lumitext("PU 200 (14 TeV)", fontsize=18, ax=ax)

        if save:
            os.makedirs(os.path.dirname(save), exist_ok=True)
            os.system(f"cp -n {php_index} {os.path.dirname(save)}")
            fig.savefig(save + ".png")
            fig.savefig(save + ".pdf")
