#%%
import sys
import os
os.makedirs("plots/step6_rate", exist_ok=True)
sys.path.append("../../../cmgrdf-cli/cmgrdf_cli/plots")

from plotters import TRate
from common import minbias
import uproot as up
import hist
import ROOT

ROOT.EnableImplicitMT()

df = ROOT.RDataFrame("Events", minbias, ["TkEleL2_pt", "TkEleL2_ptCorr", "TkEleL2_hwQual", "TkEleL2_caloEta"])

tot_events = df.Count().GetValue()

df = (df.Define("mask", "abs(TkEleL2_caloEta)<1.479")
    .Redefine("TkEleL2_pt", "TkEleL2_pt[mask]")
    .Redefine("TkEleL2_ptCorr", "TkEleL2_ptCorr[mask]")
    .Redefine("TkEleL2_hwQual", "TkEleL2_hwQual[mask]")
    .Redefine("TkEleL2_caloEta", "TkEleL2_caloEta[mask]")
    .Filter("TkEleL2_pt.size()>0")
    .Define("Lead_pt", "TkEleL2_pt[0]")
    .Redefine("TkEleL2_ptCorr", "ROOT::VecOps::Reverse(ROOT::VecOps::Sort(TkEleL2_ptCorr))")
    .Define("Lead_ptCorr", "TkEleL2_ptCorr[0]")
    .Define("tight_mask", "(TkEleL2_hwQual & 2) == 2")
)
#nevents = df.Count().GetValue()


df_tight = (
    df.Redefine("TkEleL2_pt", "TkEleL2_pt[tight_mask]")
    .Redefine("TkEleL2_ptCorr", "TkEleL2_ptCorr[tight_mask]")
    .Filter("TkEleL2_pt.size()>0")
    .Define("LeadTight_pt", "TkEleL2_pt[0]")
    .Redefine("TkEleL2_ptCorr", "ROOT::VecOps::Reverse(ROOT::VecOps::Sort(TkEleL2_ptCorr))")
    .Define("LeadTight_ptCorr", "TkEleL2_ptCorr[0]")
)
nevents_tight = df_tight.Count().GetValue()

res = df.AsNumpy(["Lead_pt", "Lead_ptCorr"])
res_tight = df_tight.AsNumpy(["LeadTight_pt", "LeadTight_ptCorr"])

#%%
def fill(array, nev):
    axis = hist.axis.Regular(100, 0, 100)
    h = hist.Hist(axis, storage=hist.storage.Weight())
    h.fill(array)
    h = h/h.integrate(0).value
    h = h * 31000 * nev/tot_events
    return h

def ptcorr_scaling(x):
    return 1.05*x+3.19

def pt_scaling(x):
    return 1.05*x+4.96


#h_pt = fill(pt_scaling(res["Lead_pt"]), nevents)
#h_pt_corr = fill(ptcorr_scaling(res["Lead_ptCorr"]), nevents)
h_pt_tight = fill(pt_scaling(res_tight["LeadTight_pt"]), nevents_tight)
h_pt_corr_tight = fill(ptcorr_scaling(res_tight["LeadTight_ptCorr"]), nevents_tight)


#%%
rate = TRate(ylim=(1, 4e4), xlim=(-1,80), xlabel = "Offline $p_{T}$ [GeV]", cmstext="Phase-2 Simulation Preliminary", lumitext="PU 200")

#rate.add(h_pt_corr, label="Regressed")
#rate.add(h_pt, label="No Regression")

rate.add(h_pt_corr_tight, label="Regressed (Tight)")
rate.add(h_pt_tight, label="No Regression (Tight)")
rate.ax.axhline(18, color="gray", linestyle="--", alpha=0.5)
#rate.save("plots/step6_rate/step6_rate_rate.pdf")
#rate.save("plots/step6_rate/step6_rate_rate.png")

#%%