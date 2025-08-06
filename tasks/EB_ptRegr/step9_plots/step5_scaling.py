#%%
import hist
import numpy as np
import matplotlib.pyplot as plt
import mplhep
import awkward as ak
import uproot as up
from scipy.optimize import curve_fit
from common import testA2
import os

os.makedirs("plots/step6_scaling", exist_ok=True)

def array(f, key):
    return ak.flatten(f[key].array()).to_numpy()


mplhep.style.use("CMS")

pt_cuts = np.linspace(12, 80, 137)

f = up.open(testA2)["Events"]

ptCorr = array(f, "TkEle_ptCorr")
pt = array(f, "TkEle_pt")
gen = array(f, "TkEle_Gen_pt")

den_h = hist.Hist(hist.axis.Regular(258, 1, 130))
den_h.fill(array(f, "TkEle_Gen_pt"))

yCorr=[]
y = []
for cut in pt_cuts:
    ptCorr_h = hist.Hist(hist.axis.Regular(258, 1, 130))
    ptCorr_h.fill(gen[ptCorr > cut])

    pt_h = hist.Hist(hist.axis.Regular(258, 1, 130))
    pt_h.fill(gen[pt > cut])

    eff_corr = ptCorr_h / den_h
    eff = pt_h / den_h

    argminCorr = np.argmin(np.abs(np.nan_to_num(eff_corr)-0.95))
    yCorr.append(den_h.axes[0].edges[argminCorr])

    argmin = np.argmin(np.abs(np.nan_to_num(eff)-0.95))
    y.append(den_h.axes[0].edges[argmin])


f = lambda x, a, b: a * x + b
poptCorr, _ = curve_fit(f, pt_cuts, yCorr)
popt, _ = curve_fit(f, pt_cuts, y)

fig, ax= plt.subplots()
ax.plot(pt_cuts, yCorr, ".", color="black", markersize=9, label="Regressed")
ax.plot(pt_cuts, y, ".", color="dodgerblue", markersize=9, label="Non-regressed")
ax.plot(pt_cuts, f(pt_cuts, *poptCorr), color="red", label=f"Regressed scaling: {poptCorr[0]:.2f} $p_T^{{L1}}$ + {poptCorr[1]:.2f} GeV", linewidth=3)

ax.plot(pt_cuts, f(pt_cuts, *popt), color="orange", label=f"Non-regressed scaling: {popt[0]:.2f} $p_T^{{L1}}$ + {popt[1]:.2f} GeV", linewidth=3)

ax.legend(fontsize=16)
ax.set_xlabel("Online pT cut [GeV]")
ax.set_ylabel("95% efficiency pT [GeV]")
mplhep.cms.text("Phase-2 Simulation Preliminary", ax=ax)
mplhep.cms.lumitext("PU 200", ax=ax)
fig.savefig("plots/step6_scaling/online_to_offline_scaling.pdf", bbox_inches="tight")
fig.savefig("plots/step6_scaling/online_to_offline_scaling.png", bbox_inches="tight")
# %%
