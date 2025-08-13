# %%
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
import matplotlib.pyplot as plt

from file_utils import openAsDataframe
from bithub.quantizers import mp_xilinx, xilinx
from common import signal_test, genpt_, pt_, ptratio_dict, q_out, features_q, scale, xgbmodel, conifermodel



#%%
xgb_model = xgb.XGBRegressor()
xgb_model.load_model(xgbmodel)


#!---------------------- HLS Model----------------------!#
cpp_cfg = conifer.backends.cpp.auto_config()
cpp_cfg["InputPrecision"] = "ap_fixed<10,1,AP_RND_CONV,AP_SAT>"
cpp_cfg["ThresholdPrecision"] = "ap_fixed<10,1,AP_RND_CONV,AP_SAT>"
cpp_cfg["ScorePrecision"] = "ap_fixed<11,2,AP_RND_CONV,AP_SAT>"

hls_model = conifer.model.load_model(conifermodel, new_config=cpp_cfg)
print("Model converted")
hls_model.compile()
print("Model compiled")

with open("conifer_conf.json", "w") as f:
    f.write(json.dumps(cpp_cfg, indent=4))

#%%
df = openAsDataframe(signal_test, "TkEle")
df = df[df[pt_]>0]
df = scale(df)

df_quant = pd.DataFrame(
    mp_xilinx.mp_xilinx(df[features_q], 'ap_fixed<10, 1, "AP_RND_CONV", "AP_SAT">', convert="double")
)
for k in features_q:
    df[k] = df_quant[k].values


# %%
xgb_full=xgb_model.predict(df[features_q])
cpp_out = hls_model.decision_function(df[features_q].values)[:,0]


diff = xgb_full - cpp_out
print(f"Max diff: {np.max(diff)}")
print(f"Min diff: {np.min(diff)}")
print(f"Mean diff: {np.mean(diff)}")


plt.hist(diff,bins=100)
plt.axvline(512*2**-9)
plt.savefig("diff_hist.png")