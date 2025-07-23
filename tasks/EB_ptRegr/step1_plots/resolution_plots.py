#%%
import sys
import numpy as np
sys.path.append("../utils")
from file_utils import openAsDataframe
from plot_utils import plot_ptratio_distributions, response_plot, plot_distributions
from compute_weights import cut_and_compute_weights

collection = "TkEle"
eta_ = f"{collection}_caloEta"
genpt_ = f"{collection}_Gen_pt"
pt_ = "TkEle_in_caloPt"
ptratio_dict = {"NoRegression": "TkEle_Gen_ptRatio"}

df = openAsDataframe("DoubleElectron_PU200.root", "TkEle")


# %%
eta_bins, centers, medians, perc5s, perc95s, perc16s, perc84s, residuals = plot_ptratio_distributions(
                            df,
                            ptratio_dict,
                            genpt_,
                            eta_,
                            plots=True
                            )
response_plot(ptratio_dict, eta_bins, centers, medians, perc5s, perc95s, perc16s, perc84s, residuals)

eta_bins, centers, medians, perc5s, perc95s, perc16s, perc84s, residuals = plot_ptratio_distributions(
                            df,
                            ptratio_dict,
                            genpt_,
                            eta_,
                            eta_bins = np.array([0,1.479])
                            #plots=True
                            )
response_plot(ptratio_dict, eta_bins, centers, medians, perc5s, perc95s, perc16s, perc84s, residuals)


# %%

df = cut_and_compute_weights(df, genpt_, pt_)
features = [
    'RESw',
    'BALw',
    'wTot',
    'TkEle_Gen_ptRatio',
    'TkEle_Gen_pt',
    'TkEle_in_caloPt',
    'TkEle_charge',
    'TkEle_idScore',
    'TkEle_in_caloLooseTkWP',
    'TkEle_in_caloRelIso',
    'TkEle_in_caloSS',
    'TkEle_in_caloStaWP',
    'TkEle_in_caloTkAbsDeta',
    'TkEle_in_caloTkAbsDphi',
    'TkEle_in_caloTkPtRatio',
    'TkEle_in_tkChi2RPhi',
    'TkEle_in_tkPtFrac',
    'TkEle_Gen_nMatch',
    'TkEle_in_caloTkNMatch',
    'TkEle_caloEta']
plot_distributions(df, features = features)



# %%
