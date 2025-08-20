# %%
import hist
import sys
import ROOT

sys.path.append("..")
from common import (
    pt_,
    genpt_,
    conifermodel,
    signal_test,
    cpp_cfg,
    features_q,
    init_pred,
)
from file_utils import open_signal
import conifer
from plotters import TEfficiency
import matplotlib.pyplot as plt
import os

basename = __file__.split("/")[-1].split(".")[0]
savefolder = f"plots/{basename}"
os.makedirs(savefolder, exist_ok=True)


df = open_signal(signal_test)
df = df[(df["TkEle_hwQual"].values & 2) == 2]

model = conifer.model.load_model(conifermodel, new_config=cpp_cfg)
model.compile()
df["ptCorr"] = (init_pred + model.decision_function(df[features_q].values)[:, 0]) * df[
    pt_
].values

den_h = hist.Hist(hist.axis.Regular(200, 3, 80))
den_h.fill(df[genpt_].values)


rdf = ROOT.RDataFrame(
    "Events",
    "/eos/cms/store/cmst3/group/l1tr/pviscone/l1teg/fp_ntuples/DoubleElectron_FlatPt-1To100_PU200/FP/151X_ptRegr_v0_A2/*.root",
    ["GenEl_caloeta", "GenEl_pt"],
)
rdf = (
    rdf.Redefine("GenEl_pt", "GenEl_pt[abs(GenEl_caloeta)<1.479]")
    .Filter("GenEl_pt.size()>0")
    .Define("GenEl_pt0", "GenEl_pt[0]")
    .Define("GenEl_pt1", "GenEl_pt[1]")
)
arr = rdf.AsNumpy(["GenEl_pt0", "GenEl_pt1"])

# %%
gen_h = hist.Hist(hist.axis.Regular(200, 3, 80))
gen_h.fill(arr["GenEl_pt0"])
gen_h.fill(arr["GenEl_pt1"])
den_h = gen_h


teff = TEfficiency(
    cmstext="Phase-2 Simulation Preliminary",
    lumitext="PU 200 (14 TeV)",
    cmstextsize=22,
    lumitextsize=22,
    xlabel="Gen $p_{T}$ [GeV]",
    rebin=5,
)
pt_h = hist.Hist(hist.axis.Regular(200, 3, 80))
pt_h.fill(df[df[pt_] > 13][genpt_].values)
ptCorr_h = hist.Hist(hist.axis.Regular(200, 3, 80))
ptCorr_h.fill(df[df["ptCorr"] > 15][genpt_].values)
teff.add(pt_h, den_h, label="No Regression ($p_{T}>13$ GeV)")
teff.add(ptCorr_h, den_h, label="Regressed ($p_{T}>15$ GeV)")
teff.add_text(3, 0.9, "Fixed Rate (200 kHz)", size=22)
teff.save(f"{savefolder}/eff200kHz.pdf")
teff.save(f"{savefolder}/eff200kHz.png")

# %%
