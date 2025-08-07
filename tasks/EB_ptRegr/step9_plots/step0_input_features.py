#%%
from common import feature_labels, features, train, feature_bins, target, target_label, target_bins, feature_map, target_map
import uproot as up
import awkward as ak
import hist
import matplotlib.pyplot as plt
import mplhep
import os
import numpy as np

os.makedirs("plots/step0_input_features/step0_input_features", exist_ok=True)

file = up.open(train)["Events"]

#%%
ylim = [
    (0, 0.9),
    (1e-5, 900),
    (0, 0.45),
    (0, 0.015),
    (1e-5, 0.9e2),
    (1e-6, 9),
]

log_scale = [
    False,
    True,
    False,
    False,
    True,
    True,
]

phys_map = [
    lambda x: x*np.pi/720,
    lambda x: x*np.pi/720,
    lambda x: x,
    lambda x: x,
    lambda x: x,
    lambda x: x
]


feature_bins = {
    "hwCaloEta": hist.axis.Regular(20, 0, 1.5),
    "in_caloTkAbsDphi": hist.axis.Regular(30, 0, 0.3),
    "in_hwTkChi2RPhi": hist.axis.Regular(30, 0, 15),
    "in_caloPt": hist.axis.Regular(20, 0, 100),
    "in_caloSS": hist.axis.Regular(30, 0.4, 1),
    "in_caloTkPtRatio": hist.axis.Regular(70, 0, 40)
}

feature_labels = {
    "hwCaloEta": r"$| \eta_{Calo} |$",
    'in_caloTkAbsDphi': r"$| \Delta \phi_{caloTk} |$",
    'in_hwTkChi2RPhi': r"$\chi^2_{tk}$ [a.u.]",
    'in_caloPt': r"$p_T^{Calo}$ [GeV]",
    'in_caloSS': r"$E_{2 \times 5}/E_{5 \times 5}$",
    'in_caloTkPtRatio': r"$p_T^{Calo}/p_T^{Tk}$"
}


#! 1d Input Features
fig, ax = plt.subplots(2, 3, figsize=(15, 10))
for i1 in range(2):
    for i2 in range(3):
        i = i1 * 3 + i2
        feat = features[i]
        data = ak.flatten(file[f"TkEle_{feat}"].array())
        lab = feature_labels[feat]
        h = hist.Hist(feature_bins[feat])
        h.fill(feature_map[feat](phys_map[i](data)))
        if i1 == 1 and i2 == 2:
            lab_leg = "DoubleElectron\n(Gen matched)"
        else:
            lab_leg = None
        mplhep.histplot(h, ax=ax[i1, i2], density=True, histtype='fill', edgecolor='black', linewidth=1.2, label=lab_leg)
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
        if i==3:
            import copy
            h_pt=copy.deepcopy(h)
ax[1,2].legend(loc='upper right', fontsize=14)
mplhep.cms.text("Phase-2 Simulation Preliminary", loc=0, ax=ax[0, 0], fontsize=20)
mplhep.cms.lumitext("PU 200", ax=ax[0, 2], fontsize=20)
fig.savefig("plots/step0_input_features/step0_input_features.pdf", bbox_inches='tight')
fig.savefig("plots/step0_input_features/step0_input_features.png", bbox_inches='tight')
#%%
#! 1d Target Feature
fig, ax = plt.subplots()
targ = ak.flatten(file[target].array())
h = hist.Hist(target_bins)
h.fill(target_map(targ))
mplhep.histplot(h, ax=ax, density=True, histtype='fill', edgecolor='black', linewidth=1.2, label="DoubleElectron\n(Gen matched)")
mplhep.histplot(h, ax=ax, density=True, color='black')
ax.set_xlabel(target_label)
ax.set_ylabel("Density")
#ax.tick_params(axis='x', labelsize=14)
#ax.tick_params(axis='y', labelsize=12)
#ax.tick_params(axis='y', which='minor', labelsize=12)
ax.set_yscale("log")
ax.legend(loc='upper right', fontsize=18)
mplhep.cms.text("Phase-2 Simulation Preliminary", loc=0, ax=ax, fontsize=22)
mplhep.cms.lumitext("PU 200", ax=ax, fontsize=22)
fig.savefig("plots/step0_input_features/step0_target_feature.pdf", bbox_inches='tight')
fig.savefig("plots/step0_input_features/step0_target_feature.png", bbox_inches='tight')

# %%


#%%
#lognorm
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

#%%
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
# %%
