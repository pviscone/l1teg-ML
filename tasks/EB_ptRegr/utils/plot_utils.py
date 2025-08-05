import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import mplhep as hep
import numpy as np
import hist
import os

hep.style.use("CMS")

colors=[
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

def plot_distributions(df, features = None, weight=None, savefolder="plots/distributions"):
    if features is None:
        keys = df.columns
    else:
        keys = features
    if isinstance(weight, str):
        weight = df[weight].values
    nfeatures = len(keys)
    os.makedirs(savefolder, exist_ok=True)
    for i2 in range(nfeatures):
        key2 = keys[i2]
        os.makedirs(os.path.join(savefolder,f"{i2}_{key2}"), exist_ok=True)
        for i1 in range(i2+1, nfeatures):
            key1 = keys[i1]
            if key1 == key2:
                continue
            fig, ax = plt.subplots()
            ax.hist2d(df[key1], df[key2], bins=(100,100), weights=weight, cmap='viridis', cmin=1, norm =LogNorm())
            ax.set_xlabel(key1)
            ax.set_ylabel(key2)
            cbar = plt.colorbar(ax.collections[0], ax=ax)
            cbar.set_label('Counts')


            fig.savefig(f"{savefolder}/{i2}_{key2}/{key1}.pdf")
            fig.savefig(f"{savefolder}/{i2}_{key2}/{key1}.png")
            plt.close(fig)



def plot_ptratio_distributions(df,
                               ptratio_dict,
                               genpt_,
                               eta_,
                               eta_bins = None,
                               genpt_bins = None,
                               plots = False,
                               savefolder = "plots"):
    os.makedirs(savefolder, exist_ok=True)
    if eta_bins is None:
        eta_bins = np.array([0, 0.7, 1.2, 1.5])
    if genpt_bins is None:
        genpt_bins = np.arange(1,105,3)
    centers = []
    medians = []
    perc5s = []
    perc16s = []
    perc84s = []
    perc95s = []
    residuals = []
    variances = []
    eta = df[eta_].values
    ptratio_dict = {key: df[value].values for key, value in ptratio_dict.items()}
    genpt = df[genpt_].values

    for eta_min, eta_max in zip(eta_bins[:-1], eta_bins[1:]):
        mask = (eta >= eta_min) & (eta < eta_max)
        genpt_eta = genpt[mask]
        centers.append({key:np.array([]) for key in ptratio_dict.keys()})
        medians.append({key:np.array([]) for key in ptratio_dict.keys()})
        perc5s.append({key:np.array([]) for key in ptratio_dict.keys()})
        perc16s.append({key:np.array([]) for key in ptratio_dict.keys()})
        perc84s.append({key:np.array([]) for key in ptratio_dict.keys()})
        perc95s.append({key:np.array([]) for key in ptratio_dict.keys()})
        residuals.append({key:np.array([]) for key in ptratio_dict.keys()})
        variances.append({key:np.array([]) for key in ptratio_dict.keys()})
        for genpt_min, genpt_max in zip(genpt_bins[:-1], genpt_bins[1:]):
            if plots:
                fig, ax = plt.subplots()
                ax.axvline(1, color='black', alpha=0.3)
                ax.set_title(f"$| \eta |$: [{eta_min},{eta_max}], GenPt: [{genpt_min},{genpt_max}]")
                ax.set_xlabel("TkEle $p_{T}$ / Gen $p_{T}$")
                ax.set_ylabel("Density")
            for idx, (label, ptratio) in enumerate(ptratio_dict.items()):
                ptratio_eta = ptratio[mask]
                mask_genpt = (genpt_eta >= genpt_min) & (genpt_eta < genpt_max)
                ptratio_masked = ptratio_eta[mask_genpt]
                median = np.median(ptratio_masked)
                perc5 = np.percentile(ptratio_masked, 5)
                perc95 = np.percentile(ptratio_masked, 95)
                perc16 = np.percentile(ptratio_masked, 16)
                perc84 = np.percentile(ptratio_masked, 84)
                res = np.median(genpt_eta[mask_genpt]*np.abs(ptratio_masked - 1))
                var = np.sum(((genpt_eta[mask_genpt]*np.abs(ptratio_masked - 1))**2)/(len(genpt_eta[mask_genpt]) - 1))
                if plots:
                    h = hist.Hist(hist.axis.Regular(29, 0.3, 1.7, name="ptratio", label="TkEle $p_{T}$ / Gen $p_{T}$"))
                    h.fill(ptratio_masked)
                    hep.histplot(h, density=True, alpha=0.75, histtype='step', label=label, linewidth=2, color=colors[idx], ax=ax)
                    ax.axvline(median, color=colors[idx], linestyle='--', label=f'Median {label}: {median:.2f}', alpha=0.7)
                    ax.axvline(perc5, color=colors[idx], linestyle=':', label=f'5% {label}: {perc5:.2f}', alpha=0.7)
                    ax.axvline(perc95, color=colors[idx], linestyle=':', label=f'95% {label}: {perc95:.2f}', alpha=0.7)

                #![Last eta bin][label]
                centers[-1][label] = np.append(centers[-1][label],((genpt_min + genpt_max) / 2))
                medians[-1][label] = np.append(medians[-1][label],(median))
                perc5s[-1][label] = np.append(perc5s[-1][label],(perc5))
                perc95s[-1][label] = np.append(perc95s[-1][label],(perc95))
                perc16s[-1][label] = np.append(perc16s[-1][label],(perc16))
                perc84s[-1][label] = np.append(perc84s[-1][label],(perc84))
                residuals[-1][label] = np.append(residuals[-1][label],(res))
                variances[-1][label] = np.append(variances[-1][label],(var))
            if plots:
                ax.legend(fontsize=15)
                fig.savefig(f"{savefolder}/ptratio_eta_{str(eta_min).replace('.','')}_{str(eta_max).replace('.','')}_genpt_{str(genpt_min).replace('.','')}_{str(genpt_max).replace('.','')}.png")
                fig.savefig(f"{savefolder}/ptratio_eta_{str(eta_min).replace('.','')}_{str(eta_max).replace('.','')}_genpt_{str(genpt_min).replace('.','')}_{str(genpt_max).replace('.','')}.pdf")
                plt.close(fig)
    return eta_bins, centers, medians, perc5s, perc95s, perc16s, perc84s, residuals, variances


def plot_ptratio_distributions_n(df,
                               ptratio_dict,
                               genpt_,
                               eta_,
                               eta_bins = None,
                               genpt_bins = None,
                               plots = False,
                               savefolder = "plots"):
    os.makedirs(savefolder, exist_ok=True)
    if eta_bins is None:
        eta_bins = np.array([0, 0.7, 1.2, 1.5])
    if genpt_bins is None:
        genpt_bins = np.arange(1,105,3)
    centers = []
    medians = []
    perc5s = []
    perc16s = []
    perc84s = []
    perc95s = []
    residuals = []
    variances = []
    n = []
    eta = df[eta_].values
    ptratio_dict = {key: df[value].values for key, value in ptratio_dict.items()}
    genpt = df[genpt_].values

    for eta_min, eta_max in zip(eta_bins[:-1], eta_bins[1:]):
        mask = (eta >= eta_min) & (eta < eta_max)
        genpt_eta = genpt[mask]
        centers.append({key:np.array([]) for key in ptratio_dict.keys()})
        medians.append({key:np.array([]) for key in ptratio_dict.keys()})
        perc5s.append({key:np.array([]) for key in ptratio_dict.keys()})
        perc16s.append({key:np.array([]) for key in ptratio_dict.keys()})
        perc84s.append({key:np.array([]) for key in ptratio_dict.keys()})
        perc95s.append({key:np.array([]) for key in ptratio_dict.keys()})
        residuals.append({key:np.array([]) for key in ptratio_dict.keys()})
        variances.append({key:np.array([]) for key in ptratio_dict.keys()})
        n.append({key:np.array([]) for key in ptratio_dict.keys()})
        for genpt_min, genpt_max in zip(genpt_bins[:-1], genpt_bins[1:]):
            if plots:
                fig, ax = plt.subplots()
                ax.axvline(1, color='black', alpha=0.3)
                ax.set_title(f"$| \eta |$: [{eta_min},{eta_max}], GenPt: [{genpt_min},{genpt_max}]")
                ax.set_xlabel("TkEle $p_{T}$ / Gen $p_{T}$")
                ax.set_ylabel("Density")
            for idx, (label, ptratio) in enumerate(ptratio_dict.items()):
                ptratio_eta = ptratio[mask]
                mask_genpt = (genpt_eta >= genpt_min) & (genpt_eta < genpt_max)
                ptratio_masked = ptratio_eta[mask_genpt]
                median = np.median(ptratio_masked)
                perc5 = np.percentile(ptratio_masked, 5)
                perc95 = np.percentile(ptratio_masked, 95)
                perc16 = np.percentile(ptratio_masked, 16)
                perc84 = np.percentile(ptratio_masked, 84)
                res = np.median(genpt_eta[mask_genpt]*np.abs(ptratio_masked - 1))
                var = np.sum(((genpt_eta[mask_genpt]*np.abs(ptratio_masked - 1))**2)/(len(genpt_eta[mask_genpt]) - 1))
                n_bins = len(ptratio_masked)
                if plots:
                    h = hist.Hist(hist.axis.Regular(29, 0.3, 1.7, name="ptratio", label="TkEle $p_{T}$ / Gen $p_{T}$"))
                    h.fill(ptratio_masked)
                    hep.histplot(h, density=True, alpha=0.75, histtype='step', label=label, linewidth=2, color=colors[idx], ax=ax)
                    ax.axvline(median, color=colors[idx], linestyle='--', label=f'Median {label}: {median:.2f}', alpha=0.7)
                    ax.axvline(perc5, color=colors[idx], linestyle=':', label=f'5% {label}: {perc5:.2f}', alpha=0.7)
                    ax.axvline(perc95, color=colors[idx], linestyle=':', label=f'95% {label}: {perc95:.2f}', alpha=0.7)

                #![Last eta bin][label]
                centers[-1][label] = np.append(centers[-1][label],((genpt_min + genpt_max) / 2))
                medians[-1][label] = np.append(medians[-1][label],(median))
                perc5s[-1][label] = np.append(perc5s[-1][label],(perc5))
                perc95s[-1][label] = np.append(perc95s[-1][label],(perc95))
                perc16s[-1][label] = np.append(perc16s[-1][label],(perc16))
                perc84s[-1][label] = np.append(perc84s[-1][label],(perc84))
                residuals[-1][label] = np.append(residuals[-1][label],(res))
                variances[-1][label] = np.append(variances[-1][label],(var))
                n[-1][label] = np.append(n[-1][label],(n_bins))

            if plots:
                ax.legend(fontsize=15)
                fig.savefig(f"{savefolder}/ptratio_eta_{str(eta_min).replace('.','')}_{str(eta_max).replace('.','')}_genpt_{str(genpt_min).replace('.','')}_{str(genpt_max).replace('.','')}.png")
                fig.savefig(f"{savefolder}/ptratio_eta_{str(eta_min).replace('.','')}_{str(eta_max).replace('.','')}_genpt_{str(genpt_min).replace('.','')}_{str(genpt_max).replace('.','')}.pdf")
                plt.close(fig)
    return eta_bins, centers, medians, perc5s, perc95s, perc16s, perc84s, residuals, variances, n


def response_plot(ptratio_dict, eta_bins, centers, medians, perc5s, perc95s, perc16s, perc84s, residuals, variances, verbose=True, savefolder="plots"):
    os.makedirs(savefolder, exist_ok=True)

    for eta_idx, (eta_min, eta_max) in enumerate(zip(eta_bins[:-1], eta_bins[1:])):
        if verbose:
            fig, ax = plt.subplots(3,1, figsize=(8, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1, 1], "hspace": 0.})
        else:
            fig, ax = plt.subplots()
            ax=[ax]
        if verbose:
            plt.setp(ax[0].get_xticklabels(), visible=False)
            plt.setp(ax[1].get_xticklabels(), visible=False)
        ax[0].axhline(1, color='gray', linestyle='-', alpha=0.5, zorder=99)
        ax[0].set_title(f"$| \eta |$: [{eta_min},{eta_max}]")
        #ax[0].set_xlabel("Gen $p_{T}$ [GeV]")
        ax[0].set_ylabel("Median TkEle $p_{T}$ / Gen $p_{T}$")
        for idx, label in enumerate(ptratio_dict.keys()):
            diff = centers[eta_idx][label][1:]-centers[eta_idx][label][:-1]
            diff = np.append(diff, diff[-1])
            #ax[0].plot(centers[-1][label], medians[-1][label], marker='o', label=label, color=colors[idx])
            ax[0].errorbar(centers[eta_idx][label]+idx*diff*0.25-diff/4, medians[eta_idx][label],
                        yerr=[medians[eta_idx][label] - perc5s[eta_idx][label], perc95s[eta_idx][label] - medians[eta_idx][label]],
                        color=colors[idx], alpha=0.7, label=f"{label} 5/95%", fmt = "o")
            ax[0].errorbar(centers[eta_idx][label]+idx*diff*0.25-diff/4, medians[eta_idx][label],
                        yerr=[medians[eta_idx][label] - perc16s[eta_idx][label], perc84s[eta_idx][label] - medians[eta_idx][label]],
                        color=colors[idx], alpha=1, label=f"{label} 16/84%", linewidth=3, markeredgecolor='black', markeredgewidth=1, markersize=5, marker='o', fmt = "o")
            for ii in range(len(centers[eta_idx][label])):
                ax[0].axvline(centers[eta_idx][label][ii]-diff[ii]/2, alpha=0.05, color="gray", linestyle=':')
            if verbose:
                ax[1].step(centers[eta_idx][label], residuals[eta_idx][label], color=colors[idx], where = "mid", alpha=0.5)
                ax[2].step(centers[eta_idx][label], variances[eta_idx][label], color=colors[idx], where = "mid", alpha=0.5)
        ax[0].legend(fontsize=18, loc='lower right')
        ax[0].set_ylim(0.3,1.7)
        if verbose:
            ax[1].set_ylim(0, 2.9)
            ax[2].set_ylim(0, 90)
            ax[1].set_ylabel("Med[|L1 $p_{T}$-Gen $p_{T}$|]", fontsize=10)
            ax[2].set_xlabel("Gen $p_{T}$ [GeV]")
            ax[2].set_ylabel(r"$\sum \frac{\left( L1 p_{T}-Gen p_{T} \right)^2]}{N-1}$", fontsize=13)
        else:
            ax[0].set_xlabel("Gen $p_{T}$ [GeV]")
        hep.cms.text("Phase-2 Simulation Preliminary", ax=ax[0], loc=2)
        hep.cms.lumitext("PU 200", ax=ax[0])
        fig.savefig(f"{savefolder}/aresponse_eta_{str(eta_min).replace('.','')}_{str(eta_max).replace('.','')}.pdf")
        fig.savefig(f"{savefolder}/aresponse_eta_{str(eta_min).replace('.','')}_{str(eta_max).replace('.','')}.png")
        plt.close(fig)


def resolution_plot(ptratio_dict, eta_bins, centers, perc16s, perc84s, variances=None, n=None, savefolder="plots"):
    os.makedirs(savefolder, exist_ok=True)

    for eta_idx, (eta_min, eta_max) in enumerate(zip(eta_bins[:-1], eta_bins[1:])):
        fig, ax = plt.subplots()
        ax.axhline(0, color='black', linestyle=':', alpha=0.3)
        ax.set_title(f"$| \eta |$: [{eta_min},{eta_max}]")
        ax.set_xlabel("Gen $p_{T}$ [GeV]")
        ax.set_ylabel("$\sigma_{eff}$(TkEle $p_{T}$ / Gen $p_{T}$)")
        for idx, label in enumerate(ptratio_dict.keys()):
            width = (perc84s[eta_idx][label] - perc16s[eta_idx][label])# / centers[eta_idx][label]
            diff=(centers[eta_idx][label][1:]-centers[eta_idx][label][:-1])/2
            if variances is not None and n is not None:
                yerr = np.sqrt(3.715* variances[eta_idx][label]/n[eta_idx][label])[:-1]/centers[eta_idx][label][:-1]
            else:
                yerr=None
            ax.errorbar(centers[eta_idx][label][:-1], width[:-1], xerr=diff, yerr=yerr, marker='o', label=label, color=colors[idx], markeredgecolor='black', markeredgewidth=1, markersize=5, ls="none")
        ax.legend()
        ax.set_ylim(0, 0.75)
        #ax.set_yscale("log")
        hep.cms.text("Phase-2 Simulation Preliminary", ax=ax, loc=2)
        hep.cms.lumitext("PU 200", ax=ax)
        fig.savefig(f"{savefolder}/resolution_eta_{str(eta_min).replace('.','')}_{str(eta_max).replace('.','')}.pdf")
        fig.savefig(f"{savefolder}/resolution_eta_{str(eta_min).replace('.','')}_{str(eta_max).replace('.','')}.png")
        plt.close(fig)