import numpy as np
import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import mplhep as hep
import hist

hep.set_style("CMS")

php_index = os.path.join(os.path.dirname(__file__), "index.php")

def _profile(df, ax=None, figsize = (8, 10), save=None):

    if ax is None:
        fig, ax = plt.subplots(figsize = figsize)

    feature_names = list(df.columns)

    # Initialize base position for the y-axis
    base_position = 0

    # Store y-tick positions and labels
    yticks_positions = []
    yticks_labels = []

    for feature_name in feature_names:
        ax.axhline(base_position + 1.05, color="black", alpha=0.2, linestyle="--", lw=1)
        ax.axhline(base_position - 1.05, color="black", alpha=0.2, linestyle="--", lw=1)

        ax.boxplot(
            df[feature_name].replace(0, np.nan).dropna().abs(),
            positions=[base_position],
            widths=1.1,
            vert=False,
            patch_artist=True,
            boxprops={"facecolor": "dodgerblue"},
            whiskerprops={
                "alpha": 1,
                "color": "black",
                "linestyle": "-",
                "linewidth": 4,
            },
            capprops={"color": "black", "linestyle": "-", "linewidth": 3,},
            medianprops={"linewidth":3},
            notch=False,
            whis=[0, 100],
            showfliers=False,
        )

        ax.boxplot(
            df[feature_name].replace(0, np.nan).dropna().abs(),
            positions=[base_position],
            widths=0.9,
            vert=False,
            patch_artist=True,
            boxprops={"alpha": 0},
            whiskerprops={"linewidth":2, "color": "white", "linestyle": "--"},
            capprops={"linewidth":3, "color": "grey"},
            medianprops={"alpha":0},
            notch=False,
            whis=[2.5, 97.5],
            showfliers=False,
        )

        # Update y-ticks positions and labels
        yticks_positions.append(base_position)
        yticks_labels.append(feature_name.replace("TkEle_", ""))

        # Increment base position for the next entry
        base_position += 2.1  # Adjust spacing between histograms

    ax.set_xlim(0.5*df.replace(0, np.nan).dropna().abs().min().min(), 2*df.replace(0, np.nan).dropna().abs().max().max())

    ax.set_yticks(yticks_positions)
    ax.set_yticklabels(yticks_labels)

    ax.set_xscale("log", base=2)
    ax.grid(True)

    hep.cms.text("Preliminary", ax=ax, fontsize=18)
    hep.cms.lumitext(df.attrs["name"], ax=ax, fontsize=18)

    if save:
        fig = plt.gcf()
        os.makedirs(os.path.dirname(save), exist_ok=True)
        os.system(f"cp -n {php_index} {os.path.dirname(save)}")
        fig.savefig(save.format(name=f"{df.attrs['name']}.pdf"), bbox_inches="tight")
        fig.savefig(save.format(name=f"{df.attrs['name']}.png"), bbox_inches="tight")

    return ax


def profile(*dfs, features=None, figsize=(8,10), save=None):
    for df in dfs:
        if features is not None:
            df = df[features]
        _profile(df, figsize = figsize, save = save)



def plot_input_features(sig_df, bkg_df, features = None, weight = None,  save=None):
    feat_info = {
        "TkEle_CryClu_pt": (r"$p_T^{\text{Cluster}}$ [GeV]", np.linspace(0,100,20)),
        "TkEle_CryClu_showerShape": (r"$E^{\text{Cluster}}_{2\times5}/E^{\text{Cluster}}_{5\times5}$", np.linspace(0,1,30)),
        "TkEle_CryClu_relIso": (r"Cluster Iso./$p^{\text{Cluster}}_T$", np.linspace(0,1.5,20)),
        "TkEle_CryClu_standaloneWP": (r"Cluster StandaloneWP", np.linspace(0,2,3)),
        "TkEle_CryClu_looseL1TkMatchWP": (r"Cluster LooseL1TkMatchWP", np.linspace(0,2,3)),
        "TkEle_Tk_chi2RPhi": (r"Tk $\chi^2_{\text{R-}\phi}$", np.linspace(0,10,20)),
        "TkEle_Tk_ptFrac": (r"$p_T^{\text{Tk}}/\sum p_T^{\text{Matched Tk}}$", np.linspace(0,1,20)),
        "TkEle_absdeta": (r"$|\Delta \eta|$ (Tk-Cluster)", np.linspace(0,0.03,20)),
        "TkEle_absdphi": (r"$|\Delta \phi|$ (Tk-Cluster)", np.linspace(0,0.3,20)),
        "TkEle_nTkMatch": (r"$N_{\text{Matched Tracks}}$", np.linspace(0,12,13)),
        "TkEle_PtRatio": (r"$p_T^{\text{Tk}}/p_T^{\text{Cluster}}$", np.linspace(0,7,20)),
    }
    n_col = min(4, len(features))
    n_row = len(features) // n_col + 1
    fig, axs = plt.subplots(n_row, n_col, figsize=(6*n_col, 6*n_row), clip_on=True)

    if len(features) == 1:
        axs = np.array([axs])
    axs = axs.flatten()

    hep.cms.text("Phase-2 Simulation Preliminary", ax=axs[0], fontsize=24)
    hep.cms.lumitext("PU 200 (14 TeV)", ax=axs[min(3, len(features))], fontsize=24)
    for idx, feat in enumerate(features):
        ax = axs[idx]

        label = feat_info[feat][0]
        bins = feat_info[feat][1]



        sig_h=hist.Hist(hist.axis.Variable(bins), storage=hist.storage.Weight())
        bkg_h=hist.Hist(hist.axis.Variable(bins), storage=hist.storage.Weight())

        sig_h.fill(sig_df[feat], weight=sig_df[weight] if weight else None)
        bkg_h.fill(bkg_df[feat], weight=bkg_df[weight] if weight else None)

        sig_h.plot(
            ax=ax,
            label="Signal",
            color="darkorange",
            linewidth=3,
            density=True,
        )
        bkg_h.plot(
            ax=ax,
            label="Background",
            color="dodgerblue",
            linewidth=3,
            density=True,
        )

        ax.set_xlabel(label, fontsize=18)
        ax.set_ylabel("a.u.") if idx % n_col == 0 else None

        if feat not in ["TkEle_CryClu_standaloneWP", "TkEle_CryClu_looseL1TkMatchWP"]:
            ax.set_yscale("log")

        if feat.startswith("TkEle_CryClu_"):
            ax.patch.set_facecolor("lightgreen")
            ax.patch.set_alpha(0.25)
        elif feat.startswith("TkEle_Tk_"):
            ax.patch.set_facecolor("lightblue")
            ax.patch.set_alpha(0.25)
        else:
            ax.patch.set_facecolor("salmon")
            ax.patch.set_alpha(0.25)

    axs[-1].axis("off")

    signal = Line2D([0], [0], color="darkorange", label=sig_df.attrs["name"], linewidth=3)
    background = Line2D([0], [0], color="dodgerblue", label=bkg_df.attrs["name"], linewidth=3)
    clu_patch = mpatches.Patch(facecolor="lightgreen", alpha=1, label="ECAL cluster", edgecolor="black")
    tk_patch = mpatches.Patch(facecolor="lightblue", alpha=1, label="L1 track", edgecolor="black")
    match_patch = mpatches.Patch(facecolor="salmon", alpha=1, label="Calo-Tk matching", edgecolor="black")
    leg1 = plt.legend(
        handles=[clu_patch, tk_patch, match_patch], loc="lower center", fontsize=22, frameon=True, title="Feature type"
    )
    plt.legend(handles=[signal, background], loc="upper left", fontsize=18)
    plt.gca().add_artist(leg1)
    if save:
        os.makedirs(os.path.dirname(save), exist_ok=True)
        os.system(f"cp -n {php_index} {os.path.dirname(save)}")
        plt.savefig(f"{save}.pdf")
        plt.savefig(f"{save}.png")
