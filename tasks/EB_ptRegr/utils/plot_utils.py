import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import hist
import os

hep.style.use("CMS")

median_colors=["red", "orange", "yellow"]
perc_colors=["green", "blue", "purple"]

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
        genpt_bins = np.linspace(1, 101, 51)
    centers = []
    medians = []
    perc5s = []
    perc95s = []
    eta = df[eta_].values
    ptratio_dict = {key: df[value].values for key, value in ptratio_dict.items()}
    genpt = df[genpt_].values

    for eta_min, eta_max in zip(eta_bins[:-1], eta_bins[1:]):
        mask = (eta >= eta_min) & (eta < eta_max)
        genpt_eta = genpt[mask]
        centers.append({key:np.array([]) for key in ptratio_dict.keys()})
        medians.append({key:np.array([]) for key in ptratio_dict.keys()})
        perc5s.append({key:np.array([]) for key in ptratio_dict.keys()})
        perc95s.append({key:np.array([]) for key in ptratio_dict.keys()})
        for genpt_min, genpt_max in zip(genpt_bins[:-1], genpt_bins[1:]):
            if plots:
                fig, ax = plt.subplots()
                ax.axvline(1, color='black', linestyle=':', alpha=0.3)
                ax.set_title(f"Eta: [{eta_min},{eta_max}], GenPt: [{genpt_min},{genpt_max}]")
                ax.set_xlabel("TkEle $p_{T}$ / Gen $p_{T}$")
                ax.set_ylabel("Density")
            for idx, (label, ptratio) in enumerate(ptratio_dict.items()):
                ptratio_eta = ptratio[mask]
                mask_genpt = (genpt_eta >= genpt_min) & (genpt_eta < genpt_max)
                ptratio_masked = ptratio_eta[mask_genpt]
                median = np.median(ptratio_masked)
                perc5 = np.percentile(ptratio_masked, 5)
                perc95 = np.percentile(ptratio_masked, 95)
                if plots:
                    h = hist.Hist(hist.axis.Regular(50, 0.3, 1.7, name="ptratio", label="TkEle $p_{T}$ / Gen $p_{T}$"))
                    h.fill(ptratio_masked)
                    hep.histplot(h, density=True, alpha=0.75, histtype='step', label=label, linewidth=2, ax=ax)
                    ax.axvline(median, color=median_colors[idx], linestyle='--', label=f'Median {label}: {median:.2f}', alpha=0.7)
                    ax.axvline(perc5, color=perc_colors[idx], linestyle='--', label=f'5% {label}: {perc5:.2f}', alpha=0.7)
                    ax.axvline(perc95, color=perc_colors[idx], linestyle='--', label=f'95% {label}: {perc95:.2f}', alpha=0.7)

                #![Last eta bin][label]
                centers[-1][label] = np.append(centers[-1][label],((genpt_min + genpt_max) / 2))
                medians[-1][label] = np.append(medians[-1][label],(median))
                perc5s[-1][label] = np.append(perc5s[-1][label],(perc5))
                perc95s[-1][label] = np.append(perc95s[-1][label],(perc95))
            if plots:
                ax.legend(fontsize=15)
                fig.savefig(f"{savefolder}/ptratio_eta_{str(eta_min).replace('.','')}_{str(eta_max).replace('.','')}_genpt_{str(genpt_min).replace('.','')}_{str(genpt_max).replace('.','')}.png")
                fig.savefig(f"{savefolder}/ptratio_eta_{str(eta_min).replace('.','')}_{str(eta_max).replace('.','')}_genpt_{str(genpt_min).replace('.','')}_{str(genpt_max).replace('.','')}.pdf")
                plt.close(fig)
    return eta_bins, centers, medians, perc5s, perc95s


def response_plot(ptratio_dict, eta_bins, centers, medians, perc5s, perc95s, savefolder="plots"):
    os.makedirs(savefolder, exist_ok=True)
    colors = ["red", "blue"]

    for eta_idx, (eta_min, eta_max) in enumerate(zip(eta_bins[:-1], eta_bins[1:])):
        fig, ax = plt.subplots()
        ax.axhline(1, color='black', linestyle=':', alpha=0.3)
        ax.set_title(f"Eta: [{eta_min},{eta_max}]")
        ax.set_xlabel("Gen $p_{T}$ [GeV]")
        ax.set_ylabel("Median TkEle $p_{T}$ / Gen $p_{T}$")
        for idx, label in enumerate(ptratio_dict.keys()):
            #ax.plot(centers[-1][label], medians[-1][label], marker='o', label=label, color=median_colors[idx])
            ax.errorbar(centers[eta_idx][label], medians[eta_idx][label],
                        yerr=[medians[eta_idx][label] - perc5s[eta_idx][label], perc95s[eta_idx][label] - medians[eta_idx][label]],
                        color=colors[idx], alpha=0.5, label=f"{label} 5/95%", marker='o', linestyle='-')
        ax.legend(fontsize=15)
        fig.savefig(f"{savefolder}/aresponse_eta_{str(eta_min).replace('.','')}_{str(eta_max).replace('.','')}.pdf")
        fig.savefig(f"{savefolder}/aresponse_eta_{str(eta_min).replace('.','')}_{str(eta_max).replace('.','')}.png")
        plt.close(fig)