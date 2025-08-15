#%%
import sys
import os
os.makedirs("plots", exist_ok=True)
sys.path.append("..")
sys.path.append("../step6_BDT_XeeScouting/flows")
from base_flow import declare
from common import bkg_test, quant, q_out, conifermodel, features_q, init_pred
from plotters import TRate

import hist
import ROOT

basename = __file__.split("/")[-1].split(".")[0]
savefolder = f"plots/{basename}"
os.makedirs(savefolder, exist_ok=True)

ROOT.EnableImplicitMT()
declare(conifermodel, (quant,1), q_out)

df = ROOT.RDataFrame("Events", bkg_test)

df_loose=(df.Define("mask", "TkEleL2_in_caloPt>0" )
    .Filter("Sum(mask)>0")
    .Define("TkEleL2_rescaled_idScore", "TkEleL2_idScore[mask]")
    .Define("TkEleL2_rescaled_caloEta", "-1 + abs(TkEleL2_caloEta[mask])")
    .Define("TkEleL2_rescaled_caloTkAbsDphi", "-1 + TkEleL2_in_caloTkAbsDphi[mask]/pow(2,5)")
    .Define("TkEleL2_rescaled_hwTkChi2RPhi", "-1 + TkEleL2_in_hwTkChi2RPhi[mask]/pow(2,3)")
    .Define("TkEleL2_rescaled_caloPt", "-1 + (TkEleL2_in_caloPt[mask])/pow(2,5)")
    .Define("TkEleL2_rescaled_caloSS", "-1 + TkEleL2_in_caloSS[mask]*2")
    .Define("TkEleL2_rescaled_caloTkPtRatio", "-1 + TkEleL2_in_caloTkPtRatio[mask]/pow(2,3)")
    .Define("out", f"( {init_pred} + bdt_evaluate({{ {','.join([f'TkEleL2_rescaled_{f}' for f in features_q])} }}))")
    .Define("pt", "TkEleL2_pt[mask]")
    .Define("ptCorr", "TkEleL2_pt[mask] * out")
    .Define("LeadPt", "Max(pt)")
    .Define("LeadPtCorr", "Max(ptCorr)")
)


df_tight = (df_loose.Define("tightMask", "(TkEleL2_hwQual[mask] & 2) == 2")
            .Filter("Sum(tightMask)>0")
            .Define("LeadPtTight", "Max(pt[tightMask])")
            .Define("LeadPtCorrTight", "Max(ptCorr[tightMask])")
        )

#%%
tot_events = df.Count().GetValue()
nev_loose = df_loose.Count().GetValue()
nev_tight = df_tight.Count().GetValue()

arr= df_tight.AsNumpy(columns=["LeadPtTight", "LeadPtCorrTight"])


def fill(array, nev):
    axis = hist.axis.Regular(100, 0, 100)
    h = hist.Hist(axis, storage=hist.storage.Weight())
    h.fill(array)
    h = h/h.integrate(0).value
    h = h * 31000 * nev/tot_events
    return h


def pt_scaling(x):
    return 1.07*x+3.84

def ptcorr_scaling(x):
    return 1.05*x+2.31


h_pt_tight_on = fill(arr["LeadPtTight"], nev_tight)
h_pt_corr_tight_on = fill(arr["LeadPtCorrTight"], nev_tight)
h_pt_tight_off = fill(pt_scaling(arr["LeadPtTight"]), nev_tight)
h_pt_corr_tight_off = fill(ptcorr_scaling(arr["LeadPtCorrTight"]), nev_tight)


#%%
rate = TRate(ylim=(1, 4e4), xlim=(-1,80), xlabel = "Offline $p_{T}$ [GeV]", cmstext="Phase-2 Simulation Preliminary", lumitext="PU 200 (14 TeV)", cmstextsize=22, lumitextsize=22)
rate.add(h_pt_corr_tight_off, label="Regressed (Tight WP)")
rate.add(h_pt_tight_off, label="No Regression (Tight WP)")
rate.ax.axhline(18, color="gray", linestyle="--", alpha=0.5)
rate.save(f"{savefolder}/offline_rate.pdf")
rate.save(f"{savefolder}/offline_rate.png")


#%%
rate = TRate(ylim=(1, 4e4), xlim=(-1,80), xlabel = "Online $p_{T}$ [GeV]", cmstext="Phase-2 Simulation Preliminary", lumitext="PU 200 (14 TeV)", cmstextsize=22, lumitextsize=22)
rate.add(h_pt_corr_tight_on, label="Regressed (Tight WP)")
rate.add(h_pt_tight_on, label="No Regression (Tight WP)")
rate.ax.axhline(18, color="gray", linestyle="--", alpha=0.5)
rate.save(f"{savefolder}/online_rate.pdf")
rate.save(f"{savefolder}/online_rate.png")
