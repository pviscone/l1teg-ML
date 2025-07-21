import uproot
import numpy as np
import matplotlib.pyplot as plt
import mplhep
import os

from math import ceil, floor, sqrt
from array import array

mplhep.style.use("CMS")


def quantiles(ys):
    if len(ys)<3:
        return (0,0,0)
    ys = ys[ys>0]
    ys = np.sort(ys)
    ny = len(ys)
    median = ys[ny//2]
   #if ny > 400e9:
   #    u95 = ys[min(int(ceil(ny*0.975)),ny-1) ]
   #    l95 = ys[int(floor(ny*0.025))]
   #    u68 = 0.5*(median+u95)
   #    l68 = 0.5*(median+l95)
    if ny > 20:
        u68 = ys[min(int(ceil(ny*0.84)),ny-1)]-median
        l68 = median-ys[int(floor(ny*0.16))]
    else:
        rms = sqrt(np.sum((y-median)**2)/ny)
        u68 = rms
        l68 = rms
    return (median,l68,u68)

eta_map=[
        ("eta_bin1", (0., 0.07)),
        ("eta_bin2", (0.07, 1.2)),
        ("eta_bin3", (1.2, 1.6))
    ]
pt_bins = np.linspace(1,100,30)
centers = (pt_bins[:-1]+pt_bins[1:])/2


folder = "/eos/user/p/pviscone/www/calibs/calibBestPhotonExtended"
corr = uproot.open(os.path.join(folder, "emcorr_barrel.root"))
tree = uproot.open("/eos/cms/store/cmst3/group/l1tr/pviscone/l1teg/fp_perfTuple/perfTuple_ElePhotons_PU0.root")["ntuple"]["tree"].arrays()

pt = tree["L1RawBarrelCalo_ptbest"].to_numpy()
mc_pt = tree["mc_pt"].to_numpy()
eta = tree["mc_eta"].to_numpy()
id = tree["mc_id"].to_numpy()

corr_pt = np.zeros_like(pt)
for pdg in [11,22]:
    part_mask = (id == pdg)
    part_pt = pt[part_mask]
    part_mcpt = mc_pt[part_mask]
    part_eta = eta[part_mask]
    part_corrpt = corr_pt[part_mask]
    for map in eta_map:
        graph = corr[map[0]]
        eta_min, eta_max = map[1]
        eta_mask = np.bitwise_and(part_eta>=eta_min, part_eta<eta_max)
        pt_etabin = part_pt[eta_mask]
        part_corrpt[eta_mask] = np.interp(pt_etabin, *graph.values())
    ratio = part_corrpt/part_mcpt

    median = []
    low_err = []
    high_err = []
    for low_pt, high_pt in zip(pt_bins[:-1],pt_bins[1:]):
        ratio_bin = ratio[np.bitwise_and(part_mcpt>=low_pt, part_mcpt<high_pt)]
        mu, err_min, err_max = quantiles(ratio_bin)
        median.append(mu)
        low_err.append(err_min)
        high_err.append(err_max)
    fig, ax = plt.subplots()
    ax.set_title(f"pdgid {pdg}")
    ax.errorbar(centers, median, yerr=np.array([low_err, high_err]), marker="^")
    ax.set_xlabel("$Gen p_T$")
    ax.axhline(1., color = "red", alpha = 0.5, linestyle="--")
    ax.set_ylim(0.75,1.5)
    ax.set_ylabel("Median $p_T^{\\text{Corr. RECO}}/ p_T^{GEN}$")
    fig.savefig(f"calibs_{pdg}.pdf")

