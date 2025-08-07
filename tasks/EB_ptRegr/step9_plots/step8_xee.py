#%%
from common import xee, xee_regr, genZd
import uproot as up
import matplotlib.pyplot as plt
import mplhep as hep
import os
import hist

os.makedirs("plots/step9_xee", exist_ok=True)

hep.style.use("CMS")

print(genZd)
xee = up.open(xee)
xee_regr = up.open(xee_regr)
genZd = up.open(genZd)

masses = [5,10,15,20,30]
xee= [xee[f"XeeM{m}"].to_hist() for m in masses]
xee_regr = [xee_regr[f"XeeM{m}"].to_hist() for m in masses]
genZd = [genZd[f"XeeM{m}"].to_hist() for m in masses]

#%%
size= (1,5)
xlims = (
    #(0,3.9),
    (1,7.4),
    (4,14),
    (7,19),
    (9,26),
    (16,36)
)
rebin = [1,1,1,1,1]
plot_gen=False

fig,ax = plt.subplots(size[0], size[1], figsize=(15, 4), sharey=True, gridspec_kw={"hspace": 0., "wspace": 0})
ax= ax.reshape(size)
for i1 in range(size[0]):
    for i2 in range(size[1]):
        i = i1*size[0] + i2
        xee_h = xee[i][hist.rebin(rebin[i])]
        xee_regr_h = xee_regr[i][hist.rebin(rebin[i])]
        genZd_h = genZd[i][hist.rebin(rebin[i])]
        hep.histplot(
            xee_h,
            ax=ax[i1,i2],
            label=f"$m_{{Zd}}$ = {masses[i]} GeV",
            histtype='fill', edgecolor='black', linewidth=1.2,
            density=True
        )
        hep.histplot(
            xee_h,
            ax=ax[i1,i2],
            color="black",
            density=True,
        )


        hep.histplot(
            xee_regr_h,
            ax=ax[i1,i2],
            color="red",
            label = "After regression",
            density=True,
        )

        if plot_gen:
            hep.histplot(
                genZd_h,
                ax=ax[i1,i2],
                color="goldenrod",
                label = "Gen",
                linestyle='--',
                linewidth=2,
                density=True,
            )


        ax[i1,i2].axvline(masses[i], 0, 0.75, alpha = 0.3, color='black', linestyle='--', linewidth=1.5)
        ax[i1,i2].set_xlim(*xlims[i])
        ax[i1,i2].set_ylim(0,1.3)
        ax[i1, i2].set_xlabel(None)
        ax[i1, i2].tick_params(axis='x', labelsize=20)
        ax[i1, i2].tick_params(axis='y', labelsize=18)
        ax[i1,i2].tick_params(axis='y', which='minor', labelsize=18)

        #remove xticks label
        #ax[i1,i2].set_xticklabels([])
        #ax[i1,i2].set_yticklabels([])
        ax[i1,i2].legend(loc = "upper right", fontsize=14)

ax[0, size[1]-1].set_xlabel("$m_{ee}$ [GeV]", fontsize=18)
ax[0,0].set_ylabel("Density", fontsize=18)
hep.cms.text("Phase-2 Simulation Preliminary", loc=0, ax=ax[0, 0], fontsize=20)
hep.cms.lumitext("PU 200", ax=ax[0, size[1]-1], fontsize=20)
fig.savefig("plots/step9_xee/step9_xee.pdf", bbox_inches='tight')
fig.savefig("plots/step9_xee/step9_xee.png", bbox_inches='tight')
# %%
