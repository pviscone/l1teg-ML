import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from common import cms10

import numpy as np
from scipy.stats import norm
from scipy.optimize import root_scalar

from hist.intervals import ratio_uncertainty
import matplotlib.pyplot as plt
import mplhep
from scipy.optimize import curve_fit
import hist

def eff_func(x, par0, par1, par2, par3, par4):
    term1_x = par0 * (x - par1)

    # Equivalent of ROOT.Math.normal_cdf(par0*(x-par1), par0*par2, 0)
    term1 = norm.cdf(term1_x, loc=0, scale=par0 * par2)

    # Equivalent of ROOT.TMath.Exp(-par0*(x-par1)+par0*par0*par2*par2/2)
    exponent = -par0 * (x - par1) + par0**2 * par2**2 / 2
    term2 = np.exp(exponent)

    # Equivalent of ROOT.Math.normal_cdf(par0*(x-par1), par0*par2, par0*par0*par2*par2)
    term3 = norm.cdf(term1_x, loc=par0**2 * par2**2, scale=par0 * par2)

    return (term1 - term2 * term3) * (par3 - par4) + par4

def find_inverse(f, y_target,  xmin, xmax, *params,):
    g = lambda x, params: f(x, *params) - y_target
    sol = root_scalar(g, bracket=[xmin, xmax], args=(params,))
    if sol.converged:
        return sol.root
    else:
        raise ValueError("Could not find the inverse for the given value.")



def derive_scaling(pt_dict, gen, pt_cuts=None, bins=None, verbose=False, savefolder=None):
    os.makedirs(savefolder, exist_ok=True)
    if pt_cuts is None:
        pt_cuts = np.linspace(10, 50, 10)
    if bins is None:
        bins = (200, 3, 80)

    den_h = hist.Hist(hist.axis.Regular(200, 3, 80))
    den_h.fill(gen)
    y95 = {}
    for pt_label in pt_dict.keys():
        y95[pt_label] = np.array([])


    for _, cut in enumerate(pt_cuts):
        if verbose:
            fig, ax = plt.subplots()
        for idx, (pt_label, pt) in enumerate(pt_dict.items()):

            pt_h = hist.Hist(hist.axis.Regular(*bins))
            pt_h.fill(gen[pt > cut])

            eff = pt_h / den_h
            x = eff.axes[0].centers

            y = eff.values()
            yErr = ratio_uncertainty(pt_h.values(), den_h.values(), uncertainty_type= "efficiency")
            yErr = (yErr[0] + yErr[1]) / 2

            popt, _ = curve_fit(eff_func, x, y, p0=[1, 0, 1, 0.95, 0.05], sigma=yErr)

            y95[pt_label] = np.append(y95[pt_label], find_inverse(eff_func, 0.95, cut-10, cut+20, *popt))

            if verbose:
                ax.errorbar(eff.axes[0].centers, y, yerr=yErr, fmt="o", color=cms10[idx], markersize=5, label=f"{pt_label}", markeredgecolor="black")
                xx=np.linspace(x[0], x[-1], 100)
                ax.plot(xx,eff_func(xx, *popt), color=cms10[idx], alpha=0.35, linewidth=3, label=f"Fit {pt_label}", linestyle="--")
                ax.plot(y95[pt_label][-1], 0.95, "*", color="white", markeredgecolor=cms10[idx], markersize=20, zorder=999, )

        if verbose:
            ax.axvline(cut, color="black", linestyle="--", linewidth=1, alpha=0.5, label=f"Online $p_T$ cut: {cut:.2f} GeV")
            ax.axhline(0.95, color="black", linestyle=":", linewidth=1, alpha = 0.3)
            ax.set_xlim(cut-10, cut+20)
            ax.set_ylim(-0.05,1.09)
            ax.set_xlabel("Gen pT [GeV]")
            ax.set_ylabel("Efficiency")
            ax.legend(fontsize=14)
            mplhep.cms.text("Phase-2 Simulation Preliminary", ax=ax, fontsize=22, loc=0)
            mplhep.cms.lumitext("PU 200 (14 TeV)", ax=ax, fontsize=22)
            if savefolder:
                fig.savefig(f"{savefolder}/turn_on_{cut:.2f}.pdf", bbox_inches="tight")
                fig.savefig(f"{savefolder}/turn_on_{cut:.2f}.png", bbox_inches="tight")


    f = lambda x, a, b: a * x + b
    fig, ax= plt.subplots()
    for idx, (pt_label, _) in enumerate(pt_dict.items()):
        popt, _ = curve_fit(f, pt_cuts, y95[pt_label])
        ax.plot(pt_cuts, y95[pt_label], ".", color=cms10[idx], markersize=14, label=f"{pt_label}")
        ax.plot(pt_cuts, f(pt_cuts, *popt), color=cms10[idx],alpha=0.3, label=f"{pt_label} scaling: {popt[0]:.2f} $p_T^{{L1}}$ + {popt[1]:.2f} GeV", linewidth=3)
    ax.legend(fontsize=16)
    ax.set_xlabel("Online pT cut [GeV]")
    ax.set_ylabel("$p_T^{95\%}$ [GeV]")
    mplhep.cms.text("Phase-2 Simulation Preliminary", ax=ax, fontsize=22)
    mplhep.cms.lumitext("PU 200 (14 TeV)", ax=ax, fontsize=22)
    if savefolder:
        fig.savefig(f"{savefolder}/online_to_offline_scaling.pdf", bbox_inches="tight")
        fig.savefig(f"{savefolder}/online_to_offline_scaling.png", bbox_inches="tight")
    return pt_cuts, y95
