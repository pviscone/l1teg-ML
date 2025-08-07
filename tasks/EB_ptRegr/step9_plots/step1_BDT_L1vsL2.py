#%%
import sys
import os

os.makedirs("plots/step1_BDT_L1vsL2", exist_ok=True)
sys.path.append("../utils")

from common import testA2
import xgboost as xgb
from file_utils import openAsDataframe
import numpy as np

features = [
    "TkEle_in_caloEta",
    #'TkEle_in_caloStaWP',
    #'TkEle_in_caloTkAbsDeta',
    'TkEle_in_caloTkAbsDphi',
    'TkEle_in_tkChi2RPhi',
    #'TkEle_in_caloLooseTkWP',
    'TkEle_in_caloPt',
    #'TkEle_in_caloRelIso',
    'TkEle_in_caloSS',
    #'TkEle_in_tkPtFrac',
    #'TkEle_in_caloTkNMatch',
    'TkEle_in_caloTkPtRatio',
    #'TkEle_idScore',
]


df = openAsDataframe(testA2, "TkEle")
#df = df[df["TkEle_hwQual"] & 2 == 2]  # keep only TkEle with hwQual 2

df["TkEle_in_caloEta"] = df["TkEle_caloEta"].abs()-1
gen = df["TkEle_Gen_pt"]
ptratio = df["TkEle_Gen_ptRatio"]
eta = df["TkEle_caloEta"]
df = df[features]

modelL1 = xgb.XGBRegressor()
modelL1.load_model("models/xgboost_model_L1.json")
modelL2 = xgb.XGBRegressor()
modelL2.load_model("models/xgboost_model_L2.json")


#%%
from plot_utils import plot_ptratio_distributions_n_width, response_plot, resolution_plot  # noqa: E402
#evaluate on test set

collection = "TkEle"
eta_ = f"{collection}_caloEta"
genpt_ = f"{collection}_Gen_pt"
pt_ = f"{collection}_in_caloPt"
ptratio_dict = {"NoRegression": "TkEle_Gen_ptRatio",
                "Regressed L2 Loss": "TkEle_regressedPtRatioL2",
                "Regressed L1 Loss": "TkEle_regressedPtRatioL1",
                }

def plot_results(modelL1, modelL2, eta_bins=None, plot_distributions=False):
    df[ptratio_dict["NoRegression"]] = ptratio
    df[genpt_] = gen
    df[eta_]=eta
    df[ptratio_dict["Regressed L1 Loss"]] = modelL1.predict(df[features].values)*df[pt_].values/df[genpt_]
    df[ptratio_dict["Regressed L2 Loss"]] = modelL2.predict(df[features].values)*df[pt_].values/df[genpt_]



    eta_bins, centers, medians, perc5s, perc95s, perc16s, perc84s, residuals, variances, n, width = plot_ptratio_distributions_n_width(df,ptratio_dict,genpt_,eta_, genpt_bins=np.linspace(1,100,34), eta_bins=eta_bins, plots=plot_distributions, savefolder="plots/step1_BDT_L1vsL2")
    resolution_plot(ptratio_dict, eta_bins, centers, width, variances=variances, n=n, savefolder="plots/step1_BDT_L1vsL2")

    response_plot(ptratio_dict, eta_bins, centers, medians, perc5s, perc95s, perc16s, perc84s, residuals, variances, savefolder="plots/step1_BDT_L1vsL2", verbose=False)
    resolution_plot(ptratio_dict, eta_bins, centers, width, variances=variances, n=n, savefolder="plots/step1_BDT_L1vsL2")

plot_results(modelL1, modelL2, eta_bins = np.array([0,1.479]), plot_distributions=False)
plot_results(modelL1, modelL2, plot_distributions=False)
# %%
