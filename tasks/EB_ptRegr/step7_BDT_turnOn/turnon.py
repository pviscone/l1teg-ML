import uproot as up
import hist
import numpy as np
import matplotlib.pyplot as plt
import mplhep
import awkward as ak

def array(f, key):
    return ak.flatten(f[key].array()).to_numpy()

mplhep.style.use("CMS")

pt_cuts=[2,5,10,20,30]
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


f = up.open("DoubleElectron_PU200_Test.root")["Events"]

ptCorr=array(f,"TkEle_ptCorr")
pt=array(f,"TkEle_pt")
gen=array(f,"TkEle_Gen_pt")

den_h = hist.Hist(hist.axis.Regular(100,1,101))
den_h.fill(array(f,"TkEle_Gen_pt"))

fig,ax=plt.subplots()
for idx,cut in enumerate(pt_cuts):

    ptCorr_h = hist.Hist(hist.axis.Regular(100,1,101))
    ptCorr_h.fill(gen[ptCorr>cut])

    pt_h = hist.Hist(hist.axis.Regular(100,1,101))
    pt_h.fill(gen[pt>cut])

    mplhep.histplot(pt_h/den_h, color = cms10[idx], label=f"pt>{cut} GeV", ax=ax, linewidth=2.5)
    mplhep.histplot(ptCorr_h/den_h, color = cms10[idx], label=f"ptCorr>{cut} GeV", linestyle="--", ax=ax, linewidth=2.5)
    ax.axvline(cut,color=cms10[idx], linestyle=":", alpha=0.5, linewidth=2)

ax.legend(loc="lower right", fontsize=14)
ax.set_xlabel(r"Gen $p_{T}$")
ax.set_xlim(-1,60)
ax.set_ylabel("Efficiency")
fig.savefig("plot.pdf")


