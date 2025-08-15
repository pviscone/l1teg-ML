#%%
import sys
sys.path.append("..")
import hist
import matplotlib.pyplot as plt
import mplhep
import os
import numpy as np
from common import signal_train, bkg_train, features_q, w, pt_
from file_utils import open_signal, open_bkg, merge_signal_bkg
from labels import feature_labels, feature_map, feature_bins, target, target_map, target_label, target_bins

basename = __file__.split("/")[-1].split(".")[0]
savefolder = f"plots/{basename}/"
os.makedirs(savefolder, exist_ok=True)



#?Open dfs
df_sig = open_signal(signal_train, scale_clip=False)
df_sig = df_sig[df_sig[pt_]>1]
df_bkg = open_bkg(bkg_train, df_sig, flat_pt=True, scale_clip=False)
df_bkg = df_bkg[df_bkg[pt_]>1]
df = merge_signal_bkg(df_sig, df_bkg)



#%%
#! 1d Input Features
def plot_features(dfs, features, ylim, log_scale, w=None, save=None):
    if not isinstance(dfs, dict):
        dfs = {"NONE":dfs}
    if len(features) <= 6:
        I1 = 2
        I2 = 3
    else:
        I1 = 3
        I2 = 3
    fig, ax = plt.subplots(I1, I2, figsize=(15, 15))
    for sample_name, df in dfs.items():
        for i1 in range(I1):
            for i2 in range(I2):
                i = i1 * I2 + i2
                if i >= len(features):
                    ax[i1, i2].axis("off")
                    continue
                feat = features[i]
                data = df[feat].values
                lab = feature_labels[feat]
                if w is not None:
                    print(w)
                    weight = df[w].values
                else:
                    weight = None
                h = hist.Hist(feature_bins[feat], storage=hist.storage.Weight())
                h.fill(feature_map[feat](data), weight=weight)
                if i == len(features) - 1:
                    lab_leg = sample_name
                else:
                    lab_leg = None
                if len(dfs) == 1:
                    histtype = 'fill'
                else:
                    histtype = 'step'
                mplhep.histplot(h, ax=ax[i1, i2], density=True, histtype=histtype, linewidth=1.2, label=lab_leg)
                if len(dfs) == 1:
                    mplhep.histplot(h, ax=ax[i1, i2], density=True, color='black')
                ax[i1, i2].set_xlabel(lab, fontsize=18)
                if log_scale[i]:
                    ax[i1, i2].set_yscale("log")
                ax[i1, i2].tick_params(axis='x', labelsize=14)
                ax[i1, i2].tick_params(axis='y', labelsize=12)
                ax[i1,i2].set_ylim(ylim[i])
                ax[i1,i2].tick_params(axis='y', which='minor', labelsize=12)
                if i2 == 0:
                    ax[i1, i2].set_ylabel("Density", fontsize=18)
                ax[i1,i2].legend(loc='upper right', fontsize=14)
    mplhep.cms.text("Phase-2 Simulation Preliminary", loc=0, ax=ax[0, 0], fontsize=20)
    mplhep.cms.lumitext("PU 200 (14 TeV)", ax=ax[0, 2], fontsize=20)
    if save:
        fig.savefig(f"{savefolder}/step0_input_features_{save}.pdf", bbox_inches='tight')
        fig.savefig(f"{savefolder}/step0_input_features_{save}.png", bbox_inches='tight')

ylim = [
    (0, 4),
    (0.001, 10),
    (1e-5, 900),
    (0, 1.3),
    (0.000001, 10),
    (1e-5, 0.9e2),
    (1e-6, 9),
]

ylim_sig = ylim[1:]
ylim_sig[2]=(0,0.4)

log_scale = [
    False,
    True,
    True,
    False,
    True,
    True,
    True,
]


plot_features({"TkEle (Gen matched)":df_sig}, features_q[1:], ylim_sig, log_scale[1:], save="signal")
plot_features({"TkEle (Gen matched)":df_sig, "Background":df_bkg}, features_q, ylim, log_scale, save="all")
plot_features({"TkEle (Gen matched)":df_sig, "Background":df_bkg}, features_q, ylim, log_scale, w=w, save="all_weighted")

#%%
#! 1d Target Feature
fig, ax = plt.subplots()
targ = df[target].values
h = hist.Hist(target_bins)
h.fill(target_map(targ))
mplhep.histplot(h, ax=ax, density=True, histtype='fill', edgecolor='black', linewidth=1.2, label="TkEle (Gen matched)")
mplhep.histplot(h, ax=ax, density=True, color='black')
ax.set_xlabel(target_label)
ax.set_ylabel("Density")
#ax.tick_params(axis='x', labelsize=14)
#ax.tick_params(axis='y', labelsize=12)
#ax.tick_params(axis='y', which='minor', labelsize=12)
ax.set_yscale("log")
ax.legend(loc='upper right', fontsize=18)
mplhep.cms.text("Phase-2 Simulation Preliminary", loc=0, ax=ax, fontsize=22)
mplhep.cms.lumitext("PU 200 (14 TeV)", ax=ax, fontsize=22)
fig.savefig(f"{savefolder}/step0_target_feature.pdf", bbox_inches='tight')
fig.savefig(f"{savefolder}/step0_target_feature.png", bbox_inches='tight')

# %%


"""
#! corrFactor vs pt
if False:
    from common import minbias
    import ROOT
    from matplotlib.colors import LogNorm

    ROOT.EnableImplicitMT()
    df = ROOT.RDataFrame("Events", minbias)
    df = df.Define("tight_mask", "abs(TkEleL2_caloEta) < 1.479 && (TkEleL2_hwQual & 2)==2").Filter("Sum(tight_mask) > 0")
    df = df.Redefine("TkEleL2_pt", "TkEleL2_pt[tight_mask][0]").Redefine("TkEleL2_ptCorr", "TkEleL2_ptCorr[tight_mask][0]")
    df = df.Define("TkEleL2_ptRatio", "TkEleL2_ptCorr/TkEleL2_pt")

    arr = df.AsNumpy(["TkEleL2_ptRatio","TkEleL2_pt", "TkEleL2_ptCorr"])

    plt.hist2d(arr["TkEleL2_pt"], arr["TkEleL2_ptRatio"], bins=(100, 100), range=((0, 100), (0, 4)), cmap='viridis', cmin=0.00001, density=True, norm=LogNorm())
    plt.xlabel("Online pT")
    plt.ylabel("ptCorr/pT")
    plt.title("Background")
    plt.colorbar(label="Density")
    plt.figure()
    plt.hist2d(ak.flatten(file["TkEle_in_caloPt"].array()).to_numpy(), ak.flatten(file[target].array()).to_numpy(), bins=(100, 100), range=((0, 100), (0, 4)), cmap='viridis', cmin=0.00001, density=True, norm=LogNorm())
    plt.xlabel("Online pT")
    plt.ylabel("ptCorr/pT")
    plt.title("Signal")
    plt.colorbar(label="Density")

if False:
    #! 2d Input Features
    targ = ak.flatten(file[target].array())
    fig, ax = plt.subplots(2, 3, figsize=(15, 10))
    for i1 in range(2):
        for i2 in range(3):
            feat = features[i1 * 3 + i2]
            data = ak.flatten(file[f"TkEle_{feat}"].array())
            lab = feature_labels[feat]
            h = hist.Hist(feature_bins[feat], target_bins)
            h.fill(feature_map[feat](data), target_map(targ))
            mplhep.hist2dplot(h, ax=ax[i1, i2], cmin=1)
            ax[i1, i2].set_xlabel(lab, fontsize=18)
            ax[i1, i2].tick_params(axis='x', labelsize=14)
            ax[i1, i2].tick_params(axis='y', labelsize=12)
            ax[i1,i2].tick_params(axis='y', which='minor', labelsize=12)
            if i2 == 0:
                ax[i1, i2].set_ylabel(target_label, fontsize=18)
    mplhep.cms.text("Phase-2 Simulation Preliminary", loc=0, ax=ax[0, 0], fontsize=20)
    mplhep.cms.lumitext("PU 200", ax=ax[0, 2], fontsize=20)
"""