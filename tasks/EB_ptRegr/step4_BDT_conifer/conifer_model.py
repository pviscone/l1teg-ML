#%%
import sys
import os
sys.path.append("../utils")
sys.path.append("..")
sys.path.append("../../../utils/BitHub")
sys.path.append("../../../utils/conifer")
sys.path = ["../utils/xgboost/python-package"] + sys.path
os.environ["PATH"] = "/data2/Xilinx/Vivado/2024.2/bin:/data2/Xilinx/Vitis_HLS/2024.2/bin:" + os.environ["PATH"]

import conifer
import xgboost as xgb
import numpy as np

from bithub.quantizers import xilinx
from plot_utils import plot_results
from common import signal_test, eta_, genpt_, pt_, ptratio_dict, metric, quant, q_out, features_q, xgbmodel, conifermodel, xilinx_cfg
from file_utils import open_signal, quantize_features

xgb_model = xgb.XGBRegressor()
xgb_model.load_model(xgbmodel)

#%%
#!----------------------Convert Model----------------------!#
hls_model = conifer.converters.convert_from_xgboost(xgb_model, xilinx_cfg)
print("Model converted")
hls_model.compile()
print("Model compiled")
hls_model.build()
print("Model built")

hls_model.save(conifermodel)


#%%
df = open_signal(signal_test)
df = quantize_features(df, features_q, quant)


#%%
ptratio_dict["Regressed (Quantized)"] = "TkEle_regressedPtRatioQuantized"

df[ptratio_dict["Regressed"]] = xilinx.convert(xilinx.ap_ufixed(q_out[0], q_out[1], q_mode = "AP_RND_CONV")(
        xgb_model.predict(df[features_q])),
    "double")* df[pt_].values/ df[genpt_].values

df[ptratio_dict["Regressed (Quantized)"]] = xilinx.convert(
    xilinx.ap_ufixed(q_out[0], q_out[1], q_mode = "AP_RND_CONV")(
        (1+hls_model.decision_function(df[features_q].values))), "double"
) * df[pt_].values / df[genpt_].values


plot_results(df, ptratio_dict, genpt_, eta_, savefolder=f"plots/plots{metric}_q{quant}_out{q_out[0]}_{q_out[1]}")
plot_results(df, ptratio_dict, genpt_, eta_, savefolder=f"plots/plots{metric}_q{quant}_out{q_out[0]}_{q_out[1]}", eta_bins=np.array([0,1.479]))
