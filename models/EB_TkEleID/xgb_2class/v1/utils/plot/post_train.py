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
        plt.savefig(f"{save}_averagegain.pdf", bbox_inches="tight")
        plt.savefig(f"{save}_averagegain.png", bbox_inches="tight")
    xgb.plot_importance(model, importance_type="weight", show_values=False)

    if save:
        plt.savefig(f"{save}_weight.pdf", bbox_inches="tight")
        plt.savefig(f"{save}_weight.png", bbox_inches="tight")

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
        plt.savefig(f"{save}_gain.pdf", bbox_inches="tight")
        plt.savefig(f"{save}_gain.png", bbox_inches="tight")
    return sorted_rank


def plot_loss(eval_result, loss="logloss", save=False):
    fig, ax = plt.subplots()
    ax.plot(eval_result["train"][loss], label="train")
    ax.plot(eval_result["eval"][loss], label="eval")
    # ax.set_yscale("log")
    ax.set_xlabel("Boosting Round")
    ax.set_ylabel("LogLoss")
    ax.legend()

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
    y=None,
    bins=np.linspace(0, 1, 30),
    save=False,
    log=False,
    func=lambda x: x,
):
    fig, ax = plt.subplots()

    classes = np.unique(df_train[y])

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

        pred_train = df_train[score][df_train[y] == cls]

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
            pred_test = df_test[score][df_test[y] == cls]

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
    return ax


def plot_roc(*dfs, label=None, score=None, y=None, weight=None, save=None, ax=None, xlim=None, ylim=None, lines=(1,0.9)):

    if ax is None:
        fig, ax = plt.subplots()

    if label is not None:
        assert len(label) == len(dfs), (
            "Number of legends must match number of dataframes"
        )
    else:
        label = [""] * len(dfs)

    aucs=[]
    for df, lab in zip(dfs, label, strict=False):
        fpr, tpr, _ = roc_curve(
            df[y], df[score], sample_weight=df[weight] if weight else None
        )
        auc = roc_auc_score(
            df[y], df[score], sample_weight=df[weight] if weight else None
        )

        aucs.append(auc)

        if lab:
            lab = f"{lab} (AUC = {auc:.2f})"
        elif df.attrs.get("name", False):
            lab = f"{df.attrs['name']} (AUC = {auc:.2f})"
        else:
            lab = f"AUC = {auc:.2f}"
        ax.plot(fpr, tpr, label=lab)

        ax.set_xlabel("Background Efficiency")
        ax.set_ylabel("Signal Efficiency")
        ax.legend(fontsize=18)

        if lines is not None:
            if not isinstance(lines, list|tuple):
                lines = [lines]
            for line in lines:
                ax.plot(line, line, color="black", linestyle="--", linewidth=0.5, alpha=0.1)

        hep.cms.text("Phase-2 Simulation Preliminary", fontsize=18, ax=ax)
        hep.cms.lumitext("PU 200 (14 TeV)", fontsize=18, ax=ax)

        if xlim:
            ax.set_xlim(xlim)

        if ylim:
            ax.set_ylim(ylim)

        if save:
            os.makedirs(os.path.dirname(save), exist_ok=True)
            os.system(f"cp -n {php_index} {os.path.dirname(save)}")
            fig.savefig(save + ".png")
            fig.savefig(save + ".pdf")
    return ax, aucs

def plot_roc_bins(df, label=None, units=None, var_name=None, var_bins=None, save = None, **kwargs):
    fig, ax = plt.subplots()
    dfs=[]
    labels=[]
    for i in range(len(var_bins)-1):
        labels.append(f"{label} = [{var_bins[i]},{var_bins[i+1]}] {units}")
        dfs.append(df[(df[var_name] > var_bins[i]) & (df[var_name] < var_bins[i+1])])
    ax, aucs = plot_roc(*dfs, ax=ax, label=labels, **kwargs)
    if save:
        os.makedirs(os.path.dirname(save), exist_ok=True)
        os.system(f"cp -n {php_index} {os.path.dirname(save)}")
        fig.savefig(save + ".png")
        fig.savefig(save + ".pdf")
    return ax, aucs

def plot_thresholds(model, weight_with_gain=False):
    model_df = model.trees_to_dataframe()
    feats = np.unique(model_df["Feature"])
    res = {}
    weights={}
    for feat in feats:
        if feat == "Leaf":
            continue
        res[feat]=model_df["Split"][model_df["Feature"]==feat].to_numpy()
        weights[feat]=model_df["Gain"][model_df["Feature"]==feat].to_numpy()
    for k in res:
        plt.hist(res[k], label = k, histtype = "step", weights=weights[k] if weight_with_gain else None, density=True)

    plt.legend()
    plt.yscale("log")


def plot_quant_aucs(quant_aucs, pt_bin_edges, save=None):
    fig, ax = plt.subplots()
    x_ticks = list(quant_aucs.keys())
    x = np.arange(len(x_ticks))

    for idx, pt_low in enumerate(pt_bin_edges[:-1]):
        pt_high = pt_bin_edges[idx+1]
        y=[]
        for q in quant_aucs.keys():
            y.append(quant_aucs[q][idx])
        ax.plot(x, y, label=f"$p_T=[{pt_low}, {pt_high}]$ GeV")


    ax.set_xticks(x)
    ax.set_xticklabels(x_ticks)
    ax.set_xlabel("bits")
    ax.set_ylabel("AUC")
    ax.legend()
    if save:
        os.makedirs(os.path.dirname(save), exist_ok=True)
        os.system(f"cp -n {php_index} {os.path.dirname(save)}")
        fig.savefig(save+".png")
        fig.savefig(save+".pdf")