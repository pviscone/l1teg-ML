import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import mplhep as hep
import numpy as np
import hist
import os
from metrics import smallest_interval_68

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

markers = ["s", "P", "X"]

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
    eta = np.abs(df[eta_].values)
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
    eta = np.abs(df[eta_].values)
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
    legend_order = []
    counter=0
    for i in range(2*len(ptratio_dict)):
        if i%2==0:
            legend_order.append(counter)
        else:
            legend_order.append(counter+len(ptratio_dict))
            counter+=1
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
        ax[0].text(0, 1.3, f"${eta_min}< | \eta | < {eta_max}$")
        #ax[0].set_xlabel("Gen $p_{T}$ [GeV]")
        ax[0].set_ylabel("$p_{T}^{\\text{TkEle}}$ / $p_{T}^{\\text{GenEle}}$")
        for idx, label in enumerate(ptratio_dict.keys()):
            diff = centers[eta_idx][label][1:]-centers[eta_idx][label][:-1]
            diff = np.append(diff, diff[-1])
            median_label=label.replace(' ',r'\ ')
            ax[0].plot(centers[eta_idx][label]+idx*diff*0.3-diff/4, medians[-1][label], ".", marker=markers[idx], label=f"$\\bf{{{median_label}}}$\nMedian", color=colors[idx], markeredgecolor='black', markeredgewidth=1, markersize=9, zorder=10)
            """
            ax[0].errorbar(centers[eta_idx][label]+idx*diff*0.3-diff/4, medians[eta_idx][label],
                                    yerr=[medians[eta_idx][label] - perc16s[eta_idx][label], perc84s[eta_idx][label] - medians[eta_idx][label]],
                                    color=colors[idx], alpha=1, label=f"{label} 16/84%", linewidth=3, fmt="o", markersize=0)
            """
            ax[0].errorbar(centers[eta_idx][label]+idx*diff*0.3-diff/4, medians[eta_idx][label],
                        yerr=[medians[eta_idx][label] - perc5s[eta_idx][label], perc95s[eta_idx][label] - medians[eta_idx][label]],
                        color=colors[idx], alpha=1, label="5/95% Quantiles", fmt = "o", markersize=0, linewidth=2)

            for ii in range(len(centers[eta_idx][label])):
                ax[0].axvline(centers[eta_idx][label][ii]-diff[ii]/2, alpha=0.05, color="gray", linestyle=':')
            if verbose:
                ax[1].step(centers[eta_idx][label], residuals[eta_idx][label], color=colors[idx], where = "mid", alpha=0.5)
                ax[2].step(centers[eta_idx][label], variances[eta_idx][label], color=colors[idx], where = "mid", alpha=0.5)
        handles, labels = ax[0].get_legend_handles_labels()
        ax[0].legend([handles[idx] for idx in legend_order],
                     [labels[idx] for idx in legend_order],
                     fontsize=20, loc='lower right')
        #ax[0].legend()
        ax[0].set_ylim(0.3,1.4)
        if verbose:
            ax[1].set_ylim(0, 2.9)
            ax[2].set_ylim(0, 90)
            ax[1].set_ylabel("Med[|L1 $p_{T}$-Gen $p_{T}$|]", fontsize=10)
            ax[2].set_xlabel("Gen $p_{T}$ [GeV]")
            ax[2].set_ylabel(r"$\sum \frac{\left( L1 p_{T}-Gen p_{T} \right)^2]}{N-1}$", fontsize=13)
        else:
            ax[0].set_xlabel("Gen $p_{T}$ [GeV]")
        hep.cms.text("Phase-2 Simulation Preliminary", ax=ax[0], loc=0, fontsize=22)
        hep.cms.lumitext("PU 200 (14 TeV)", ax=ax[0], fontsize=22)
        fig.savefig(f"{savefolder}/aresponse_eta_{str(eta_min).replace('.','')}_{str(eta_max).replace('.','')}.pdf")
        fig.savefig(f"{savefolder}/aresponse_eta_{str(eta_min).replace('.','')}_{str(eta_max).replace('.','')}.png")
        plt.close(fig)

import ROOT
def convert_to_th1(boost_hist, name, title):
    # Ensure the input is a 1D histogram
    if len(boost_hist.axes) != 1:
        raise ValueError("This function only supports 1D histograms.")

    # Get histogram properties from the boost histogram
    n_bins = boost_hist.axes[0].size
    x_min = boost_hist.axes[0].edges[0]
    x_max = boost_hist.axes[0].edges[-1]

    # Create the ROOT TH1D object
    root_hist = ROOT.TH1D(name, title, n_bins, x_min, x_max)

    # Get the view for efficient data access
    view = boost_hist.values()

    # Loop through the bins and transfer data
    for i in range(n_bins):
        # ROOT bins are 1-indexed, so we use i + 1
        bin_content = view[i]
        bin_error = np.sqrt(boost_hist.variances()[i])
        root_hist.SetBinContent(i + 1, bin_content)
        root_hist.SetBinError(i + 1, bin_error)

    # Transfer entries
    root_hist.SetEntries(boost_hist.sum())

    return root_hist

def effSigma(hist):
    xaxis = hist.GetXaxis()
    nb = xaxis.GetNbins()
    if nb < 10:
        print("effsigma: Not a valid histo. nbins = {}".format(nb))
        return -1

    bwid = xaxis.GetBinWidth(1)
    if bwid == 0:
        print("effsigma: Not a valid histo. bwid = {}".format(bwid))
        return -1

    xmax = xaxis.GetXmax()
    xmin = xaxis.GetXmin()
    ave = hist.GetMean()
    rms = hist.GetRMS()

#     print 'xmax: {}, xmin: {}, ave: {}, rms: {}'.format(xmax, xmin, ave, rms)

    total = 0.
    for i in range(0, nb+2):
        total += hist.GetBinContent(i)

    if total < 100.:
        print("effsigma: Too few entries {}".format(total))
        return -1

    ierr = 0
    ismin = 999

    rlim = 0.683*total
    # Set scan size to +/- rms
    nrms = int(rms/(bwid))
    if nrms > nb/10:
        # could be tuned
        nrms = nb/10

    widmin = 9999999.
    # scan the window center
    for iscan in range(int(-nrms), int(nrms)+1):
        ibm = int((ave-xmin)/bwid+1+iscan)
        x = (ibm-0.5) * bwid + xmin
        xj = x
        xk = x
        jbm = ibm
        kbm = ibm
        bin = hist.GetBinContent(ibm)
        total = bin
        for j in range(1, nb):
            if jbm < nb:
                jbm += 1
                xj += bwid
                bin = hist.GetBinContent(jbm)
                total += bin
                if total > rlim:
                    break
            else:
                ierr = 1
            if kbm > 0:
                kbm -= 1
                xk -= bwid
                bin = hist.GetBinContent(kbm)
                total += bin
                if total > rlim:
                    break
            else:
                ierr = 1
        dxf = (total-rlim)*bwid/bin
        wid = (xj-xk+bwid-dxf)*0.5
        if wid < widmin:
            widmin = wid
            ismin = iscan

    if ismin == nrms or ismin == -nrms:
        ierr = 3
    if ierr != 0:
        print("effsigma: Error of type {}".format(ierr))

    return widmin



def plot_ptratio_distributions_n_width(df,
                               ptratio_dict,
                               genpt_,
                               eta_,
                               eta_bins = None,
                               genpt_bins = None,
                               plots = False,
                               savefolder = "plots"):
    os.makedirs(savefolder, exist_ok=True)
    if eta_bins is None:
        eta_bins = np.array([0, 0.7, 1.2, 1.479])
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
    width = []
    n = []
    eta = np.abs(df[eta_].values)
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
        width.append({key:np.array([]) for key in ptratio_dict.keys()})
        n.append({key:np.array([]) for key in ptratio_dict.keys()})
        for genpt_min, genpt_max in zip(genpt_bins[:-1], genpt_bins[1:]):
            if plots:
                fig, ax = plt.subplots()
                ax.axvline(1, color='black', alpha=0.3)
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
                ptratio_hist = hist.Hist(hist.axis.Regular(50, 0.35, 1.6))
                ptratio_hist.fill(ptratio_masked)
                ptratio_hist = convert_to_th1(ptratio_hist, f"ptratio_{label}_eta_{eta_min}_{eta_max}_genpt_{genpt_min}_{genpt_max}", f"TkEle pT / Gen pT {label} for $| \eta |$: [{eta_min},{eta_max}], GenPt: [{genpt_min},{genpt_max}]")
                wid = effSigma(ptratio_hist)

                n_bins = len(ptratio_masked)
                if plots:
                    h = hist.Hist(hist.axis.Regular(29, 0.3, 1.7))
                    h.fill(ptratio_masked)
                    hep.histplot(h, density=True, alpha=1, histtype='step', linewidth=3, color=colors[idx], ax=ax)
                    ax.axvline(median, color=colors[idx], linestyle='--', label=f'Median: {median:.2f}', alpha=0.7)
                    eff_sigma = smallest_interval_68(ptratio_masked)
                    ax.fill_betweenx([0, 10], eff_sigma[0], eff_sigma[1], color=colors[idx], alpha=0.15, label=r'$\sigma_{\text{eff}}$',zorder=-999)
                    #ax.axvline(perc5, color=colors[idx], linestyle=':', label=f'5% {label}: {perc5:.2f}', alpha=0.7)
                    #ax.axvline(perc95, color=colors[idx], linestyle=':', label=f'95% {label}: {perc95:.2f}', alpha=0.7)

                #![Last eta bin][label]
                centers[-1][label] = np.append(centers[-1][label],((genpt_min + genpt_max) / 2))
                medians[-1][label] = np.append(medians[-1][label],(median))
                perc5s[-1][label] = np.append(perc5s[-1][label],(perc5))
                perc95s[-1][label] = np.append(perc95s[-1][label],(perc95))
                perc16s[-1][label] = np.append(perc16s[-1][label],(perc16))
                perc84s[-1][label] = np.append(perc84s[-1][label],(perc84))
                residuals[-1][label] = np.append(residuals[-1][label],(res))
                variances[-1][label] = np.append(variances[-1][label],(var))
                width[-1][label] = np.append(width[-1][label],(wid))
                n[-1][label] = np.append(n[-1][label],(n_bins))

            if plots:
                ax.set_ylim(0, 10)
                ax.set_xlim(0.2, 1.8)
                ax.text(0.25, 8.5, f"${eta_min} < | \eta | < {eta_max} $\n${genpt_min} < p_T^{{\\text{{Gen}}}} < {genpt_max}$ GeV", fontsize=22)
                ax.set_ylabel("Density")
                ax.set_xlabel(r"$p_{T}^{\text{TkEle}}$ / $p_{T}^{\text{Gen}}$")
                ax.legend(fontsize=22, loc='upper right')
                hep.cms.text("Phase-2 Simulation Preliminary", ax=ax, loc=0, fontsize=22)
                hep.cms.lumitext("PU 200 (14 TeV)", ax=ax, fontsize=22)
                fig.savefig(f"{savefolder}/ptratio_eta_{str(eta_min).replace('.','')}_{str(eta_max).replace('.','')}_genpt_{str(genpt_min).replace('.','')}_{str(genpt_max).replace('.','')}.png")
                fig.savefig(f"{savefolder}/ptratio_eta_{str(eta_min).replace('.','')}_{str(eta_max).replace('.','')}_genpt_{str(genpt_min).replace('.','')}_{str(genpt_max).replace('.','')}.pdf")
                plt.close(fig)
    return eta_bins, centers, medians, perc5s, perc95s, perc16s, perc84s, residuals, variances, n, width



def resolution_plot(ptratio_dict, eta_bins, centers, width, variances=None, n=None, savefolder="plots"):
    os.makedirs(savefolder, exist_ok=True)

    for eta_idx, (eta_min, eta_max) in enumerate(zip(eta_bins[:-1], eta_bins[1:])):
        fig, ax = plt.subplots()
        ax.axhline(0, color='black', linestyle=':', alpha=0.3)
        ax.text(0, 0.28, f"${eta_min} < | \eta | < {eta_max}$")
        ax.set_xlabel("Gen $p_{T}$ [GeV]")
        ax.set_ylabel("$\sigma_{eff}$($p_{T}^{\\text{TkEle}}$ / $p_{T}^{\\text{GenEle}}$)")
        for idx, label in enumerate(ptratio_dict.keys()):
            #width = (perc84s[eta_idx][label] - perc16s[eta_idx][label])# / centers[eta_idx][label]
            diff=(centers[eta_idx][label][1:]-centers[eta_idx][label][:-1])/2
            if variances is not None and n is not None:
                yerr = np.sqrt(3.715* variances[eta_idx][label]/n[eta_idx][label])[:-1]/centers[eta_idx][label][:-1]
            else:
                yerr=None
            ax.errorbar(centers[eta_idx][label][:-1], width[eta_idx][label][:-1], xerr=diff, yerr=yerr/2, marker='o', label=label, color=colors[idx], markeredgecolor='black', markeredgewidth=1, markersize=5, ls="none")
        ax.legend()
        ax.set_ylim(0, 0.3)
        #ax.set_yscale("log")
        hep.cms.text("Phase-2 Simulation Preliminary", ax=ax, loc=0, fontsize=22)
        hep.cms.lumitext("PU 200 (14 TeV)", ax=ax, fontsize=22)
        fig.savefig(f"{savefolder}/resolution_eta_{str(eta_min).replace('.','')}_{str(eta_max).replace('.','')}.pdf")
        fig.savefig(f"{savefolder}/resolution_eta_{str(eta_min).replace('.','')}_{str(eta_max).replace('.','')}.png")
        plt.close(fig)

def plot_xgb_loss(l1_metric, savefolder="plots"):
    fig, ax = plt.subplots()
    for k, metric in l1_metric.items():
        if k=="train":
            linestyle = "-"
        else:
            linestyle = "--"
        ax.plot(metric.sig_mae*25, label=f"25*Signal MAE ({k})", linestyle=linestyle,color="red")
        ax.plot(metric.sig_median, label=f"Sig median ({k})", linestyle=linestyle,color="purple")
        ax.plot(metric.sig_quant95, label=f"Sig 95% quantile ({k})", linestyle=linestyle,color="orange")
        ax.plot(metric.bkg_mae, label=f"Bkg Mean ({k})", linestyle=linestyle,color="blue")
        ax.plot(metric.bkg_quant95, label=f"Bkg 95% quantile ({k})", linestyle=linestyle,color="green")
        #ax.plot(metric.mae, label=f"Total MAE ({k})", linestyle=linestyle,color="orange")
    ax.axhline(1, color='black', linestyle='--', alpha = 0.5)
    ax.legend(fontsize = 12, loc = "center right")
    hep.cms.text("Phase-2 Simulation Preliminary")
    hep.cms.lumitext("PU 200")
    if savefolder:
        os.makedirs(savefolder, exist_ok=True)
        fig.savefig(savefolder + "/metrics.png")
        fig.savefig(savefolder + "/metrics.pdf")
    plt.show()


def plot_xgb_importance(model, features, savefolder="plots"):
    # Plot feature importance
    importances = model.feature_importances_
    plt.figure(figsize=(10, 6))
    plt.barh(features, importances)
    plt.xlabel("Feature Importance")
    plt.title("XGBoost Feature Importances")
    plt.tight_layout()
    if savefolder:
        os.makedirs(savefolder, exist_ok=True)
        plt.savefig(savefolder + "/feature_importance.pdf")
        plt.savefig(savefolder + "/feature_importance.png")
    plt.show()



def plot_results(df, ptratio_dict, genpt_, eta_, eta_bins=None, verbose=False, genpt_bins=None, savefolder="plots"):
    if genpt_bins is None:
        genpt_bins = np.linspace(1,100,34)
    os.makedirs(savefolder, exist_ok=True)
    eta_bins, centers, medians, perc5s, perc95s, perc16s, perc84s, residuals, variances, n, width = plot_ptratio_distributions_n_width(df,ptratio_dict,genpt_,eta_, genpt_bins=genpt_bins, eta_bins=eta_bins, plots=verbose, savefolder=savefolder)
    response_plot(ptratio_dict, eta_bins, centers, medians, perc5s, perc95s, perc16s, perc84s, residuals, variances, savefolder=savefolder, verbose=verbose)
    resolution_plot(ptratio_dict, eta_bins, centers, width, variances=variances, n=n, savefolder=savefolder)


def plot_bkg(model, df_test, features_q, pt_, verbose=False, savefolder=None, what="all", bins=None):
    if bins is None:
        bins = np.arange(1,101,5)
    if what.lower()=="all":
        what = "median,q64,q95,mean"
    what= what.lower().split(",")
    os.makedirs(savefolder, exist_ok=True)
    df_test_bkg = df_test[df_test["label"].values==0]
    df_test_sig = df_test[df_test["label"].values==1]
    median_bkg = np.array([])
    q95_bkg = np.array([])
    q64_bkg = np.array([])
    mean_bkg = np.array([])
    median_sig = np.array([])
    q95_sig = np.array([])
    q64_sig = np.array([])
    mean_sig = np.array([])
    center = np.array([])
    for min_pt, max_pt in zip(bins[:-1], bins[1:]):
        df_bkg_pt = df_test_bkg[(df_test_bkg[pt_] >= min_pt) & (df_test_bkg[pt_] < max_pt)]
        df_sig_pt = df_test_sig[(df_test_sig[pt_] >= min_pt) & (df_test_sig[pt_] < max_pt)]

        if len(df_bkg_pt) == 0:
            continue
        df_bkg_pt["model_output"] = model.predict(df_bkg_pt[features_q])
        df_sig_pt["model_output"] = model.predict(df_sig_pt[features_q])
        if verbose:
            fig, ax = plt.subplots()
            ax.hist(df_bkg_pt["model_output"].values, bins=100, range=(0, 2), density=True,label="BKG")
            ax.hist(df_sig_pt["model_output"].values, bins=100, range=(0, 2), density=True,label="SIG", histtype='step')
            ax.set_title(f"pt: {min_pt} - {max_pt}")
            ax.legend()
            ax.set_xlabel("Model output")
            ax.set_ylabel("Density")
            fig.savefig(os.path.join(savefolder, f"pt_{min_pt}_{max_pt}.png"))
        median_bkg = np.append(median_bkg,np.median(df_bkg_pt["model_output"]))
        q95_bkg= np.append(q95_bkg,np.quantile(df_bkg_pt["model_output"], 0.95))
        q64_bkg= np.append(q64_bkg,np.quantile(df_bkg_pt["model_output"], 0.64))
        mean_bkg= np.append(mean_bkg, np.mean(df_bkg_pt["model_output"]))
        median_sig = np.append(median_sig,np.median(df_sig_pt["model_output"]))
        q95_sig= np.append(q95_sig,np.quantile(df_sig_pt["model_output"], 0.95))
        q64_sig= np.append(q64_sig,np.quantile(df_sig_pt["model_output"], 0.64))
        mean_sig= np.append(mean_sig, np.mean(df_sig_pt["model_output"]))
        center = np.append(center, (min_pt + max_pt)/2)
    fig, ax = plt.subplots()
    if "median" in what:
        ax.plot(center, median_bkg, label="Bkg median", color="blue", marker='o')
        ax.plot(center, median_sig, label="Sig median", color="blue", linestyle='--', marker='o')
    if "q64" in what:
        ax.plot(center, q64_bkg, label="Bkg 64%", color="orange", marker='o')
        ax.plot(center, q64_sig, label="Sig 64%", color="orange", linestyle='--', marker='o')
    if "q95" in what:
        ax.plot(center, q95_bkg, label="Bkg 95%", color="red", marker='o')
        ax.plot(center, q95_sig, label="Sig 95%", color="red", linestyle='--', marker='o')
    if "mean" in what:
        ax.plot(center, mean_bkg, label="Bkg mean", color="green", marker='o')
        ax.plot(center, mean_sig, label="Sig mean", color="green", linestyle='--', marker='o')
    ax.set_xlabel("Calo pT")
    ax.set_ylabel("Model output")
    ax.legend()
    if savefolder is not None:
        fig.savefig(os.path.join(savefolder, "bkg_sig.png"))
        fig.savefig(os.path.join(savefolder, "bkg_sig.pdf"))