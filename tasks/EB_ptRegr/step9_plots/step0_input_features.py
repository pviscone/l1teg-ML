#%%
from common import feature_labels, features, train, feature_bins, target, target_label, target_bins, feature_map, target_map
import uproot as up
import awkward as ak
import hist
import matplotlib.pyplot as plt
import mplhep
import os

os.makedirs("plots/step0_input_features/step0_input_features", exist_ok=True)

file = up.open(train)["Events"]

#%%
#! 1d Input Features
fig, ax = plt.subplots(2, 3, figsize=(15, 10))
for i1 in range(2):
    for i2 in range(3):
        feat = features[i1 * 3 + i2]
        data = ak.flatten(file[f"TkEle_{feat}"].array())
        lab = feature_labels[feat]
        h = hist.Hist(feature_bins[feat])
        h.fill(feature_map[feat](data))
        if i1 == 1 and i2 == 2:
            lab_leg = "DoubleElectron\n(Gen matched)"
        else:
            lab_leg = None
        mplhep.histplot(h, ax=ax[i1, i2], density=True, histtype='fill', edgecolor='black', linewidth=1.2, label=lab_leg)
        mplhep.histplot(h, ax=ax[i1, i2], density=True, color='black')
        ax[i1, i2].set_xlabel(lab, fontsize=18)
        ax[i1, i2].set_yscale("log")
        ax[i1, i2].tick_params(axis='x', labelsize=14)
        ax[i1, i2].tick_params(axis='y', labelsize=12)
        ax[i1,i2].tick_params(axis='y', which='minor', labelsize=12)
        if i2 == 0:
            ax[i1, i2].set_ylabel("Density", fontsize=18)
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
