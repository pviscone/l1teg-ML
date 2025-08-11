#%%
import hist
import numpy as np
import matplotlib.pyplot as plt
import mplhep
import awkward as ak
import uproot as up
from scipy.optimize import curve_fit
from hist.intervals import ratio_uncertainty
from common import testA2
from scipy.stats import norm
import os
import sys
sys.path.append("../utils")
sys.path.append("../../../utils/BitHub")
sys.path.append("../../../utils/conifer")
from bithub.quantizers import mp_xilinx
from file_utils import openAsDataframe
import conifer
from common_q import features_q, scale, init_pred
import pandas as pd

os.makedirs("plots/step5_scaling", exist_ok=True)

def array(f, key):
    return ak.flatten(f[key].array()).to_numpy()

import ROOT
def f_yc(x, par0, par1, par2, par3, par4):
    term1_x = par0 * (x - par1)

    # Equivalent of ROOT.Math.normal_cdf(par0*(x-par1), par0*par2, 0)
    term1 = norm.cdf(term1_x, loc=0, scale=par0 * par2)

    # Equivalent of ROOT.TMath.Exp(-par0*(x-par1)+par0*par0*par2*par2/2)
    exponent = -par0 * (x - par1) + par0**2 * par2**2 / 2
    term2 = np.exp(exponent)

    # Equivalent of ROOT.Math.normal_cdf(par0*(x-par1), par0*par2, par0*par0*par2*par2)
    term3 = norm.cdf(term1_x, loc=par0**2 * par2**2, scale=par0 * par2)

    return (term1 - term2 * term3) * (par3 - par4) + par4

from scipy.optimize import root_scalar
def find_inverse(f, y_target,  xmin, xmax, *params,):
    g = lambda x, params: f(x, *params) - y_target
    sol = root_scalar(g, bracket=[xmin, xmax], args=(params,))
    if sol.converged:
        return sol.root
    else:
        raise ValueError("Could not find the inverse for the given value.")

mplhep.style.use("CMS")

pt_cuts = np.linspace(10, 50, 10)
cms10 = [
    "#3f90da",
    "#ffa90e",
    "#bd1f01",
    "#94a4a2",
    "#832db6",
    "#a96b59",
    "#e76300",
    "#b9ac70",
    "#92dadd",
    "#717581",
]

f = up.open(testA2)["Events"]

pt = array(f, "TkEle_pt")
hwQual = array(f, "TkEle_hwQual")
gen = array(f, "TkEle_Gen_pt")
#%%
cpp_cfg = conifer.backends.cpp.auto_config()
cpp_cfg["InputPrecision"] = "ap_fixed<10,1,AP_RND_CONV,AP_SAT>"
cpp_cfg["ThresholdPrecision"] = "ap_fixed<10,1,AP_RND_CONV,AP_SAT>"
cpp_cfg["ScorePrecision"] = "ap_fixed<12,3,AP_RND_CONV,AP_SAT>"

model_q = conifer.model.load_model("models/conifer_model_L1_q10_hls.json", new_config=cpp_cfg)
model_q.compile()

quant = 10
q_out = (12,3)
df = openAsDataframe(testA2, "TkEle")
df = scale(df)
df[features_q] = pd.DataFrame(
    mp_xilinx.mp_xilinx(df[features_q], f'ap_fixed<{quant}, 1, "AP_RND_CONV", "AP_SAT">', convert="double")
)
ptCorr = (init_pred+model_q.decision_function(df[features_q].values)[:,0])*pt

tight_mask = (hwQual & 2 )==2
ptCorr = ptCorr[tight_mask]
pt = pt[tight_mask]
gen = gen[tight_mask]

den_h = hist.Hist(hist.axis.Regular(200, 3, 80))
den_h.fill(gen)

y95 = []
yCorr95 = []

for idx,cut in enumerate(pt_cuts):
    fig, ax = plt.subplots()
    ptCorr_h = hist.Hist(hist.axis.Regular(200, 3, 80))
    ptCorr_h.fill(gen[ptCorr > cut])

    pt_h = hist.Hist(hist.axis.Regular(200, 3, 80))
    pt_h.fill(gen[pt > cut])

    eff_corr = ptCorr_h / den_h
    eff = pt_h / den_h

    x = eff.axes[0].centers

    yCorr = eff_corr.values()
    yCorrErr = ratio_uncertainty(ptCorr_h.values(), den_h.values(), uncertainty_type= "efficiency")
    yCorrErr = (yCorrErr[0]+ yCorrErr[1]) / 2

    y = eff.values()
    yErr = ratio_uncertainty(pt_h.values(), den_h.values(), uncertainty_type= "efficiency")
    yErr = (yErr[0] + yErr[1]) / 2

    poptCorr, _ = curve_fit(f_yc, x, yCorr, p0=[1, 0, 1, 0.95, 0.05], sigma=yCorrErr)
    popt, _ = curve_fit(f_yc, x, y, p0=[1, 0, 1, 0.95, 0.05], sigma=yErr)

    #eff.plot(ax=ax, label=f"Non-Regressed: {cut:.2f} GeV", color="red", linestyle="--")
    #eff_corr.plot(ax=ax, label=f"Regressed: {cut:.2f} GeV", color="blue", linestyle="--")

    ax.errorbar(eff.axes[0].centers, y, yerr=yErr, fmt="o", color="blue", markersize=5, label="Non-Regressed", markeredgecolor="black")
    ax.errorbar(eff_corr.axes[0].centers, yCorr, yerr=yCorrErr, fmt="o", color="red", markersize=5, label="Regressed", markeredgecolor="black")

    xx=np.linspace(x[0], x[-1], 100)
    ax.axvline(cut, color="black", linestyle="--", linewidth=1, alpha=0.5, label=f"Online pT cut: {cut:.2f} GeV")
    ax.plot(xx,f_yc(xx, *poptCorr), color="orange", linewidth=3, label="Fit Regressed", linestyle="--")
    ax.plot(xx,f_yc(xx, *popt), color="dodgerblue", linewidth=3, label="Fit Non-regressed", linestyle="--")
    ax.axhline(0.95, color="black", linestyle=":", linewidth=1, alpha = 0.3)
    ax.set_xlim(cut-10, cut+20)
    ax.set_ylim(-0.05,1.09)
    ax.legend(fontsize=14)
    y95.append(find_inverse(f_yc, 0.95, cut-10, cut+20, *popt))
    yCorr95.append(find_inverse(f_yc, 0.95, cut-10, cut+20, *poptCorr))

    ax.plot(y95[-1], 0.95, "*", color="white", markeredgecolor="dodgerblue", markersize=20, zorder=999, )
    ax.plot(yCorr95[-1], 0.95,"*", color="white", markeredgecolor="red", markersize=20, zorder=999, )
    ax.set_xlabel("Gen pT [GeV]")
    ax.set_ylabel("Efficiency pT [GeV]")
    fig.savefig(f"plots/step5_scaling/turn_on_{cut:.2f}.pdf", bbox_inches="tight")
    fig.savefig(f"plots/step5_scaling/turn_on_{cut:.2f}.png", bbox_inches="tight")

#%%


f = lambda x, a, b: a * x + b
poptCorr, _ = curve_fit(f, pt_cuts, yCorr95)
popt, _ = curve_fit(f, pt_cuts, y95)

fig, ax= plt.subplots()
ax.plot(pt_cuts, yCorr95, ".", color="black", markersize=14, label="Regressed")
ax.plot(pt_cuts, y95, ".", color="dodgerblue", markersize=14, label="Non-regressed")
ax.plot(pt_cuts, f(pt_cuts, *poptCorr), color="red", label=f"Regressed scaling: {poptCorr[0]:.2f} $p_T^{{L1}}$ + {poptCorr[1]:.2f} GeV", linewidth=3)

ax.plot(pt_cuts, f(pt_cuts, *popt), color="orange", label=f"Non-regressed scaling: {popt[0]:.2f} $p_T^{{L1}}$ + {popt[1]:.2f} GeV", linewidth=3)

ax.legend(fontsize=16)
ax.set_xlabel("Online pT cut [GeV]")
ax.set_ylabel("95% efficiency pT [GeV]")
mplhep.cms.text("Phase-2 Simulation Preliminary", ax=ax)
mplhep.cms.lumitext("PU 200", ax=ax)
fig.savefig("plots/step5_scaling/online_to_offline_scaling.pdf", bbox_inches="tight")
fig.savefig("plots/step5_scaling/online_to_offline_scaling.png", bbox_inches="tight")
# %%
