#%%
import sys
import os
sys.path.append("../utils")
sys.path.append("../../../utils/BitHub")
sys.path.append("../../../utils/conifer")

import conifer
import xgboost as xgb
import json
import numpy as np
import pandas as pd

from file_utils import openAsDataframe
from bithub.quantizers import mp_xilinx

cfg = conifer.backends.xilinxhls.auto_config(granularity="full")
cfg["XilinxPart"] = "xcvu13p-flga2577-2-e"
cfg['InputPrecision'] = "ap_fixed<10,1,AP_RND_CONV,AP_SAT>"
cfg['ThresholdPrecision'] = "ap_fixed<10,1,AP_RND_CONV,AP_SAT>"
cfg['ScorePrecision'] =  "ap_ufixed<12,3,AP_RND_CONV,AP_SAT>"
cfg['ClockPeriod'] = 4.16666666


xgb_model = xgb.XGBRegressor()
xgb_model.load_model("xgboost_model_L1_q10.json")

#%%
#!----------------------Convert Model----------------------!#
hls_model = conifer.converters.convert_from_xgboost(xgb_model, cfg)
print("Model converted")
hls_model.compile()
print("Model compiled")
hls_model.build()
print("Model built")

hls_model.save("conifer_model_L1_q10_hls.json")
with open("conifer_conf.json", "w") as f:
    f.write(json.dumps(cfg, indent=4))

#%%
#!---------------------- Validate ----------------------!#
collection = "TkEle"
eta_ = f"{collection}_caloEta"
genpt_ = f"{collection}_Gen_pt"
pt_ = f"{collection}_in_caloPt"
ptratio_dict = {"NoRegression": "TkEle_Gen_ptRatio",
                "Regressed": "TkEle_regressedPtRatio",
                "HLSRegressed": "TkEle_hls_regressedPtRatio"}

if not os.path.exists("DoubleElectron_PU200.root"):
    raise ValueError("xrdcp root://eosuser.cern.ch//eos/user/p/pviscone/www/L1T/l1teg/EB_ptRegr/step0_ntuple/DoubleEle_PU200/zsnap/era151Xv0pre4_TkElePtRegr_dev_withScaled/base_2_ptRatioMultipleMatch05/DoubleElectron_PU200.root .")


df = openAsDataframe("DoubleElectron_PU200.root", "TkEle")

features = [
    #'TkEle_in_caloStaWP',
    #'TkEle_in_caloTkAbsDeta',
    #'TkEle_in_caloLooseTkWP',
    #'TkEle_idScore',
    "hwCaloEta",
    'caloTkAbsDphi',
    'hwTkChi2RPhi',
    'caloPt',
    'caloRelIso',
    'caloSS',
    'tkPtFrac',
    'caloTkNMatch',
    'caloTkPtRatio',
]


df["hwCaloEta"] = np.clip(-1+df["TkEle_hwCaloEta"].abs()/2**8, -1, 1)
df["caloTkAbsDphi"] = np.clip(-1 + df["TkEle_in_caloTkAbsDphi"]/2**5, -1, 1)
df["hwTkChi2RPhi"] = np.clip(-1 + df["TkEle_in_hwTkChi2RPhi"]/2**3, -1, 1)
df["caloPt"] = np.clip(-1+(df["TkEle_in_caloPt"]-1)/2**6, -1, 1)
df["caloRelIso"] = np.clip(-1 + df["TkEle_in_caloRelIso"]*2**3, -1, 1)
df["caloSS"] = np.clip(-1 + df["TkEle_in_caloSS"]*2, -1, 1)
df["tkPtFrac"] = np.clip(-1 + df["TkEle_in_tkPtFrac"]/2**3, -1, 1)
df["caloTkNMatch"] = np.clip(-1 + df["TkEle_in_caloTkNMatch"]/2**2, -1, 1)
df["caloTkPtRatio"] = np.clip(-1 + df["TkEle_in_caloTkPtRatio"]/2**3, -1, 1)


df[features] = pd.DataFrame(
    mp_xilinx.mp_xilinx(df[features], 'ap_fixed<{10, 1, "AP_RND_CONV", "AP_SAT">', convert="double")
)

#%%
from plot_utils import plot_ptratio_distributions, response_plot  # noqa: E402
#evaluate on test set
def plot_results(model, hlsmodel, plot_distributions=False, savefolder="plots"):
    df[ptratio_dict["Regressed"]] = model.predict(df[features].values)*df[pt_].values/df[genpt_]

    df[ptratio_dict["HLSRegressed"]] = (1+hlsmodel.decision_function(df[features].values))*df[pt_].values/df[genpt_]

    eta_bins, centers, medians, perc5s, perc95s, perc16s, perc84s, residuals, variances = plot_ptratio_distributions(df,ptratio_dict,genpt_,eta_, genpt_bins=np.linspace(1,100,34), plots=plot_distributions, savefolder=savefolder)
    response_plot(ptratio_dict, eta_bins, centers, medians, perc5s, perc95s, perc16s, perc84s, residuals, variances, savefolder=savefolder)

plot_results(xgb_model, hls_model, plot_distributions=True, savefolder="plots")
# %%
