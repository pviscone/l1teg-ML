# %%
import uproot as up
import matplotlib.pyplot as plt
import mplhep as hep
import os
import hist
import sys

sys.path.append("..")
from common import xee_hs, xee_regr_hs, genZd_hs
from metrics import effSigmaInterval

basename = __file__.split("/")[-1].split(".")[0]
savefolder = f"plots/{basename}"
os.makedirs(savefolder, exist_ok=True)

hep.style.use("CMS")

xee = up.open(xee_hs)
xee_regr = up.open(xee_regr_hs)
genZd = up.open(genZd_hs)

masses = [5, 10, 15, 20, 30]
xee = [xee[f"XeeM{m}"].to_hist() for m in masses]
xee_regr = [xee_regr[f"XeeM{m}"].to_hist() for m in masses]
genZd = [genZd[f"XeeM{m}"].to_hist() for m in masses]

# %%
size = (1, 5)
xlims = (
    # (0,3.9),
    (1, 7.4),
    (4, 14),
    (7, 19),
    (9, 26),
    (16, 36),
)
rebin = [50, 50, 50, 50, 50]
plot_gen = False

sigma_arr = []
sigma_err_arr = []
sigmaCorr_arr = []
sigmaCorr_err_arr = []

fig, ax = plt.subplots(
    size[0],
    size[1],
    figsize=(15, 4),
    sharey=True,
    gridspec_kw={"hspace": 0.0, "wspace": 0},
)
ax = ax.reshape(size)
for i1 in range(size[0]):
    for i2 in range(size[1]):
        i = i1 * size[0] + i2
        xee_h = xee[i]
        xee_regr_h = xee_regr[i]
        genZd_h = genZd[i]

        sigma = effSigmaInterval(xee_h)
        sigma = sigma[1] - sigma[0]
        sigma_arr.append(sigma)
        hep.histplot(
            xee_h[hist.rebin(rebin[i])],
            ax=ax[i1, i2],
            label=f"$m_{{Zd}}$ = {masses[i]} GeV",
            color="#3f90da",
            histtype="fill",
            edgecolor="black",
            linewidth=1.2,
            density=True,
        )
        hep.histplot(
            xee_h[hist.rebin(rebin[i])],
            ax=ax[i1, i2],
            color="black",
            density=True,
        )

        sigma_regr = effSigmaInterval(xee_regr_h)
        sigma_regr = sigma_regr[1] - sigma_regr[0]
        sigmaCorr_arr.append(sigma_regr)
        hep.histplot(
            xee_regr_h[hist.rebin(rebin[i])],
            ax=ax[i1, i2],
            color="#ffa90e",
            label="After regression",
            density=True,
        )

        if plot_gen:
            hep.histplot(
                genZd_h[hist.rebin(rebin[i])],
                ax=ax[i1, i2],
                color="goldenrod",
                label="Gen",
                linestyle="--",
                linewidth=2,
                density=True,
            )

        ax[i1, i2].axvline(
            masses[i], 0, 0.75, alpha=0.3, color="black", linestyle="--", linewidth=1.5
        )
        ax[i1, i2].set_xlim(*xlims[i])
        ax[i1, i2].set_ylim(0, 1.5)
        ax[i1, i2].set_xlabel(None)
        ax[i1, i2].tick_params(axis="x", labelsize=20)
        ax[i1, i2].tick_params(axis="y", labelsize=18)
        ax[i1, i2].tick_params(axis="y", which="minor", labelsize=18)

        # remove xticks label
        # ax[i1,i2].set_xticklabels([])
        # ax[i1,i2].set_yticklabels([])
        ax[i1, i2].legend(loc="upper right", fontsize=14)

ax[0, size[1] - 1].set_xlabel("$m_{ee}$ [GeV]", fontsize=18)
ax[0, 0].set_ylabel("Density", fontsize=18)
hep.cms.text("Phase-2 Simulation Preliminary", loc=0, ax=ax[0, 0], fontsize=22)
hep.cms.lumitext("PU 200 (14 TeV)", ax=ax[0, size[1] - 1], fontsize=22)
fig.savefig(f"{savefolder}/step9_xee.pdf", bbox_inches="tight")
fig.savefig(f"{savefolder}/step9_xee.png", bbox_inches="tight")
# %%
import numpy as np
import mplhep

sigma_arr = np.array(sigma_arr)
sigmaCorr_arr = np.array(sigmaCorr_arr)

fig, ax = plt.subplots(
    2,
    1,
    figsize=(10, 6),
    sharex=True,
    gridspec_kw={"hspace": 0, "height_ratios": [2, 1]},
)
ax[0].plot(masses, sigma_arr, marker="o", label="Before regression")
ax[0].plot(masses, sigmaCorr_arr, marker="o", label="After regression")
ax[0].legend()
ax[0].set_ylabel("$\sigma_{\\text{eff}}(m_{Z_d})$ [GeV]")
ax[1].plot(masses, sigmaCorr_arr / sigma_arr, marker="o", color="black")
ax[1].set_xlabel("$m_{Zd}$ [GeV]")
ax[1].set_ylim(0.5, 1.7)
ax[1].set_ylabel(
    "$\\frac{\\text{After}}{\\text{Before}}$",
    fontsize=20,
)
ax[1].legend(fontsize=14)
mplhep.cms.text("Phase-2 Simulation Preliminary", loc=0, ax=ax[0], fontsize=22)
mplhep.cms.lumitext("PU 200  (14 TeV)", ax=ax[0], fontsize=22)
fig.savefig(f"{savefolder}/step9_xee_sigma.pdf", bbox_inches="tight")
fig.savefig(f"{savefolder}/step9_xee_sigma.png", bbox_inches="tight")
