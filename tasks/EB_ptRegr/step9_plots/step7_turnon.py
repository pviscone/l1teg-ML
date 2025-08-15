# %%
import hist
import sys
import ROOT
sys.path.append("..")
from common import pt_, genpt_, conifermodel, signal_test, cpp_cfg, features_q, init_pred
from file_utils import open_signal
import conifer
import matplotlib.pyplot as plt
df = open_signal(signal_test)
df = df[(df["TkEle_hwQual"].values & 2) == 2 ]

model = conifer.model.load_model(conifermodel, new_config=cpp_cfg)
model.compile()
df["ptCorr"] = (init_pred + model.decision_function(df[features_q].values)[:, 0])*df[pt_].values

den_h = hist.Hist(hist.axis.Regular(200, 3, 80))
den_h.fill(df[genpt_].values)

rdf = ROOT.RDataFrame("Events", "/eos/cms/store/cmst3/group/l1tr/pviscone/l1teg/fp_ntuples/DoubleElectron_FlatPt-1To100_PU200/FP/151X_ptRegr_v0_A2/*.root", ["GenEl_caloeta","GenEl_pt"])
rdf = rdf.Redefine("GenEl_pt", "GenEl_pt[abs(GenEl_caloeta)<1.479]").Filter("GenEl_pt.size()>0").Define("GenEl_pt0", "GenEl_pt[0]").Define("GenEl_pt1", "GenEl_pt[1]")
arr = rdf.AsNumpy(["GenEl_pt0", "GenEl_pt1"])

#%%
gen_h = hist.Hist(hist.axis.Regular(200, 3, 80))
gen_h.fill(arr["GenEl_pt0"])
gen_h.fill(arr["GenEl_pt1"])
den_h=gen_h

def pt_descaling(x):
    return (x-3.84)/1.07

def ptcorr_descaling(x):
    return (x-2.31)/1.05


fig, ax = plt.subplots()
pt_h = hist.Hist(hist.axis.Regular(200, 3, 80))
pt_h.fill(df[df[pt_].values > pt_descaling(25)][genpt_].values)

ptCorr_h = hist.Hist(hist.axis.Regular(200, 3, 80))
ptCorr_h.fill(df[df["ptCorr"].values > ptcorr_descaling(25)][genpt_].values)

print(f"online pt 25 {pt_descaling(25):.2f}")
print(f"online ptCorr 25 {ptcorr_descaling(25):.2f}")

eff = pt_h / den_h
effCorr = ptCorr_h/den_h
x = eff.axes[0].centers
ax.step(x, eff.values(), label="Non-Regressed", color="red")
ax.step(x, effCorr.values(), label="Regressed", color="blue")
ax.legend()
ax.set_xlabel("Gen pT")
# %%
