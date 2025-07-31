#%%
from cmgrdf_cli.plots.plotters import TRate
import uproot as up

pt = up.open("../temp/era151X_ptRegr_v0_A1/main/LeadTkEleL2_pt.root")["MinBias"].to_hist()
ptCorr = up.open("../temp/era151X_ptRegr_v0_A1/main/LeadTkEleL2_ptCorr.root")["MinBias"].to_hist()

pt_tight = up.open("../temp/era151X_ptRegr_v0_A1/main/LeadTkEleL2_tight_pt.root")["MinBias"].to_hist()
ptCorr_tight = up.open("../temp/era151X_ptRegr_v0_A1/main/LeadTkEleL2_tight_ptCorr.root")["MinBias"].to_hist()

#%%
rate = TRate(ylim=(1, 4e4), xlabel = "Online $p_{T}$ [GeV]")
rate.add(pt, label="pt")
rate.add(ptCorr, label="ptCorr")

rate.save("rate_ptCorr.pdf")
rate.save("rate_ptCorr.png")
#%%
rate_tight = TRate(ylim=(1, 4e4), xlabel = "Online $p_{T}$ [GeV]")
rate_tight.add(pt_tight, label="pt (Tight)")
rate_tight.add(ptCorr_tight, label="ptCorr (Tight)")
rate_tight.save("rate_ptCorr_tight.pdf")
rate_tight.save("rate_ptCorr_tight.png")