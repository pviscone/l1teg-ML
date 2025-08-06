#%%
import sys
import os
os.makedirs("plots/step7_turnon", exist_ok=True)
sys.path.append("../../../cmgrdf-cli/cmgrdf_cli/plots")

from plotters import TEfficiency

import uproot as up
import hist
import numpy as np
import matplotlib.pyplot as plt
import mplhep
import awkward as ak
from common import testA2, doubleEle
import ROOT
ROOT.EnableImplicitMT()

def array(f, key):
    return ak.flatten(f[key].array()).to_numpy()

mplhep.style.use("CMS")



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




#%%


df = ROOT.RDataFrame("Events", doubleEle, ["GenEl_caloeta","GenEl_pt"])
df = df.Redefine("GenEl_pt", "GenEl_pt[abs(GenEl_caloeta)<1.479]").Filter("GenEl_pt.size()>0").Define("GenEl_pt0", "GenEl_pt[0]").Define("GenEl_pt1", "GenEl_pt[1]")
arr = df.AsNumpy(["GenEl_pt0", "GenEl_pt1"])

gen_h = hist.Hist(hist.axis.Regular(100, 1, 101))
gen_h.fill(arr["GenEl_pt0"])
gen_h.fill(arr["GenEl_pt1"])

f = up.open(testA2)["Events"]

ptCorr=array(f,"TkEle_ptCorr")
pt=array(f,"TkEle_pt")
gen=array(f,"TkEle_Gen_pt")

tight_mask = array(f, "TkEle_hwQual") & 2 == 2

den_h = hist.Hist(hist.axis.Regular(100,1,101))
den_h.fill(array(f,"TkEle_Gen_pt"))
#%%
cuts = {
    "tight":
        (37, 40),
    "loose":
        (55, 63)
}

def ptcorr_scaling(x):
    return 1.07*x+2.02

def pt_scaling(x):
    return 1.07*x+4.23


def plot_eff(den, lab, cuts, text=None):
    eff = TEfficiency(legend_kwargs={"loc": "center right", "fontsize":16}, xlabel=r"Gen $p_{T} [GeV]$", xlim=(20,100), cmstext="Phase-2 Simulation Preliminary", lumitext="PU 200", grid = False, ylim=(-0.05, 1.1))

    if "loose" in cuts:
        pt_h = hist.Hist(hist.axis.Regular(100,1,101))
        pt_h.fill(gen[pt_scaling(pt)>cuts["loose"][0]])

        ptCorr_h = hist.Hist(hist.axis.Regular(100,1,101))
        ptCorr_h.fill(gen[ptcorr_scaling(ptCorr)>cuts["loose"][1]])

    pt_h_tight = hist.Hist(hist.axis.Regular(100,1,101))
    pt_h_tight.fill(gen[tight_mask][pt_scaling(pt[tight_mask])>cuts["tight"][0]])

    ptCorr_h_tight = hist.Hist(hist.axis.Regular(100,1,101))
    ptCorr_h_tight.fill(gen[tight_mask][ptcorr_scaling(ptCorr[tight_mask])>cuts["tight"][1]])

    if "loose" in cuts:
        eff.add(ptCorr_h, den, label=f"Offline $p_T^{{corr}}$>{cuts['loose'][1]} GeV", linestyle="--")
        eff.add(pt_h, den, label=f"Offline $p_T$>{cuts['loose'][0]} GeV", linestyle="-")

    eff.add(ptCorr_h_tight, den, label=f"Offline $p_T^{{corr}}$>{cuts['tight'][1]} GeV (Tight)", linestyle="--")
    eff.add(pt_h_tight, den, label=f"Offline $p_T$>{cuts['tight'][0]} GeV (Tight)", linestyle="-")

    if text:
        eff.ax.text(0.05, 0.95, text, transform=eff.ax.transAxes, fontsize=20, color="black", weight="bold")
    eff.save(f"plots/step7_turnon/step7_turnon_efficiency_{lab}.pdf")
    eff.save(f"plots/step7_turnon/step7_turnon_efficiency_{lab}.png")


plot_eff(den_h, "18khz_matched", cuts, text="Fixed Rate (18 kHz)")
plot_eff(gen_h, "18khz_all", cuts, text="Fixed Rate (18 kHz)")

cuts_fixed = {
    "tight":
        (36, 36),
}
plot_eff(den_h, "36_matched", cuts_fixed)
plot_eff(gen_h, "36_all", cuts_fixed)