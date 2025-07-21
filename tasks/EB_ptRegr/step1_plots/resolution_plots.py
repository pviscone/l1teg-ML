#%%
import sys
sys.path.append("../utils")
from file_utils import openAsDataframe
from plot_utils import plot_ptratio_distributions, response_plot

collection = "TkEle"
eta_ = f"{collection}_caloEta"
genpt_ = f"{collection}_Gen_pt"
ptratio_dict = {"NoRegression": "TkEle_Gen_ptRatio"}

df = openAsDataframe("/eos/user/p/pviscone/www/L1T/l1teg/EB_ptRegr/step0_ntuple/DoubleEle_PU200/zsnap/era151Xv0pre4_TkElePtRegr_dev/base_2_ptRatioMultipleMatch05/DoubleElectron_PU200.root", "TkEle")


# %%
eta_bins, centers, medians, perc5s, perc95s= plot_ptratio_distributions(
                            df,
                            ptratio_dict,
                            genpt_,
                            eta_,)
response_plot(ptratio_dict, eta_bins, centers, medians, perc5s, perc95s)

# %%



