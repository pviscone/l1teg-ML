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
import json
import numpy as np
import pandas as pd

from file_utils import openAsDataframe
from bithub.quantizers import mp_xilinx, xilinx
from plot_utils import plot_results
from common import signal_test, eta_, genpt_, pt_, ptratio_dict, metric, quant, q_out, features_q, scale,xgbmodel, conifermodel

cfg = conifer.backends.xilinxhls.auto_config(granularity="full")
cfg["XilinxPart"] = "xcvu13p-flga2577-2-e"
cfg['InputPrecision'] = "ap_fixed<10,1,AP_RND_CONV,AP_SAT>"
cfg['ThresholdPrecision'] = "ap_fixed<10,1,AP_RND_CONV,AP_SAT>"
cfg['ScorePrecision'] =  "ap_fixed<11,2,AP_RND_CONV,AP_SAT>"
cfg['ClockPeriod'] = 4.16666666


xgb_model = xgb.XGBRegressor()
xgb_model.load_model(xgbmodel)

#%%
#!----------------------Convert Model----------------------!#
hls_model = conifer.converters.convert_from_xgboost(xgb_model, cfg)
print("Model converted")
hls_model.compile()
print("Model compiled")
hls_model.build()
print("Model built")

hls_model.save(conifermodel)
with open("conifer_conf.json", "w") as f:
    f.write(json.dumps(cfg, indent=4))

#%%
df = openAsDataframe(signal_test, "TkEle")
df = df[df[pt_]>0]
df = scale(df)

df_quant = pd.DataFrame(
    mp_xilinx.mp_xilinx(df[features_q], 'ap_fixed<10, 1, "AP_RND_CONV", "AP_SAT">', convert="double")
)
for k in features_q:
    df[k] = df_quant[k].values

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
