#%%
import sys
import os

os.makedirs("plots/step4_quantization", exist_ok=True)
sys.path.append("../utils")
sys.path.append("../../../utils/BitHub")
sys.path.append("../../../utils/conifer")

from common import testA2
from common_q import features_q, scale, init_pred

import xgboost as xgb
from file_utils import openAsDataframe
import numpy as np
import conifer
import copy


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
df["TkEle_in_caloEta"] = df["TkEle_caloEta"].abs()-1

df_q = copy.deepcopy(df)
df_q = scale(df_q)


bdt = xgb.XGBRegressor()
bdt.load_model("models/xgboost_model_L1.json")

cpp_cfg = conifer.backends.cpp.auto_config()
cpp_cfg["InputPrecision"] = "ap_fixed<10,1,AP_RND_CONV,AP_SAT>"
cpp_cfg["ThresholdPrecision"] = "ap_fixed<10,1,AP_RND_CONV,AP_SAT>"
cpp_cfg["ScorePrecision"] = "ap_fixed<12,3,AP_RND_CONV,AP_SAT>"

model_q = conifer.model.load_model("models/conifer_model_L1_q10_hls.json", new_config=cpp_cfg)
model_q.compile()


#%%
from plot_utils import plot_ptratio_distributions_n_width, response_plot, resolution_plot  # noqa: E402
#evaluate on test set

collection = "TkEle"
eta_ = f"{collection}_caloEta"
genpt_ = f"{collection}_Gen_pt"
pt_ = f"{collection}_in_caloPt"
ptratio_dict = {"NoRegression": "TkEle_Gen_ptRatio",
                "BDT": "BDT",
                "Quantized BDT": "Quantized BDT"}

def plot_results(bdt, eta_bins=None, plot_distributions=False):

    df[ptratio_dict["BDT"]] = bdt.predict(df[features].values)*df[pt_].values/df[genpt_]
    df[ptratio_dict["Quantized BDT"]] = (init_pred+model_q.decision_function(df_q[features_q].values)[:,0])*df[pt_].values/df[genpt_]

    eta_bins, centers, medians, perc5s, perc95s, perc16s, perc84s, residuals, variances, n, width = plot_ptratio_distributions_n_width(df,ptratio_dict,genpt_,eta_, genpt_bins=np.linspace(1,100,34), eta_bins=eta_bins, plots=plot_distributions, savefolder="plots/step4_quantization")
    response_plot(ptratio_dict, eta_bins, centers, medians, perc5s, perc95s, perc16s, perc84s, residuals, variances, savefolder="plots/step4_quantization", verbose=False)
    resolution_plot(ptratio_dict, eta_bins, centers, width, variances=variances, n=n, savefolder="plots/step4_quantization")

plot_results(bdt, eta_bins = np.array([0,1.479]), plot_distributions=False)
plot_results(bdt, plot_distributions=False)
# %%
