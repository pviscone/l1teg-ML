# %%
import sys
import os

sys.path.append("../utils")
sys.path.append("../../../utils/BitHub")
sys.path.append("../../../utils/conifer")
os.environ["PATH"] = (
    "/data2/Xilinx/Vivado/2024.2/bin:/data2/Xilinx/Vitis_HLS/2024.2/bin:"
    + os.environ["PATH"]
)

import conifer
import xgboost as xgb
import numpy as np
import pandas as pd

from file_utils import openAsDataframe
from bithub.quantizers import mp_xilinx

xgb_model = xgb.XGBRegressor()
xgb_model.load_model("xgboost_model_L1_q10.json")

# %%
#!----------------------Convert Model----------------------!#
cfg = conifer.backends.xilinxhls.auto_config(granularity="full")
cfg["XilinxPart"] = "xcvu13p-flga2577-2-e"
cfg["InputPrecision"] = "ap_fixed<10,1,AP_RND_CONV,AP_SAT>"
cfg["ThresholdPrecision"] = "ap_fixed<10,1,AP_RND_CONV,AP_SAT>"
cfg["ScorePrecision"] = "ap_fixed<12,3,AP_RND_CONV,AP_SAT>"
cfg["ClockPeriod"] = 4.16666666



hls_model = conifer.converters.convert_from_xgboost(xgb_model, cfg)
print("Model converted")
hls_model.compile()
print("Model compiled")
#hls_model.build()
#print("Model built")

# %%
#!---------------------- Validate ----------------------!#
collection = "TkEle"
eta_ = f"{collection}_caloEta"
genpt_ = f"{collection}_Gen_pt"
pt_ = f"{collection}_in_caloPt"
ptratio_dict = {
    "NoRegression": "TkEle_Gen_ptRatio",
    "Regressed": "TkEle_regressedPtRatio",
    "HLSRegressed": "TkEle_hls_regressedPtRatio",
}

if not os.path.exists("DoubleElectron_PU200.root"):
    raise ValueError(
        "xrdcp root://eosuser.cern.ch//eos/user/p/pviscone/www/L1T/l1teg/EB_ptRegr/step0_ntuple/DoubleEle_PU200/zsnap/era151Xv0pre4_TkElePtRegr_dev_withScaled/base_2_ptRatioMultipleMatch05/DoubleElectron_PU200.root ."
    )


df = openAsDataframe("DoubleElectron_PU200.root", "TkEle")

features = [
    #'TkEle_in_caloStaWP',
    #'TkEle_in_caloTkAbsDeta',
    #'TkEle_in_caloLooseTkWP',
    #'TkEle_idScore',
    "hwCaloEta",
    "caloTkAbsDphi",
    "hwTkChi2RPhi",
    "caloPt",
    #"caloRelIso",
    "caloSS",
    "tkPtFrac",
    "caloTkNMatch",
    "caloTkPtRatio",
]


df["hwCaloEta"] = np.clip(-1 + df["TkEle_hwCaloEta"].abs() / 2**8, -1, 1)
df["caloTkAbsDphi"] = np.clip(-1 + df["TkEle_in_caloTkAbsDphi"] / 2**5, -1, 1)
df["hwTkChi2RPhi"] = np.clip(-1 + df["TkEle_in_hwTkChi2RPhi"] / 2**3, -1, 1)
df["caloPt"] = np.clip(-1 + (df["TkEle_in_caloPt"] - 1) / 2**6, -1, 1)
df["caloRelIso"] = np.clip(-1 + df["TkEle_in_caloRelIso"] * 2**3, -1, 1)
df["caloSS"] = np.clip(-1 + df["TkEle_in_caloSS"] * 2, -1, 1)
df["tkPtFrac"] = np.clip(-1 + df["TkEle_in_tkPtFrac"] / 2**3, -1, 1)
df["caloTkNMatch"] = np.clip(-1 + df["TkEle_in_caloTkNMatch"] / 2**2, -1, 1)
df["caloTkPtRatio"] = np.clip(-1 + df["TkEle_in_caloTkPtRatio"] / 2**3, -1, 1)


df[features] = pd.DataFrame(
    mp_xilinx.mp_xilinx(
        df[features], 'ap_fixed<10, 1, "AP_RND_CONV", "AP_SAT">', convert="double"
    )
)

# %%
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
cpp_cfg = conifer.backends.cpp.auto_config()
cpp_cfg["InputPrecision"] = "ap_fixed<10,1,AP_RND_CONV,AP_SAT>"
cpp_cfg["ThresholdPrecision"] = "ap_fixed<10,1,AP_RND_CONV,AP_SAT>"
cpp_cfg["ScorePrecision"] = "ap_fixed<12,3,AP_RND_CONV,AP_SAT>"

cpp_model = conifer.converters.convert_from_xgboost(xgb_model, cpp_cfg)
cpp_model.compile()
# %%
xgb_full=xgb_model.predict(df[features])
hls_out = hls_model.decision_function(df[features].values)
cpp_out = cpp_model.decision_function(df[features].values)[:,0]

# %%
diff = xgb_full - cpp_out
print(f"Max diff: {np.max(diff)}")
print(f"Min diff: {np.min(diff)}")
print(f"Mean diff: {np.mean(diff)}")


# %%
