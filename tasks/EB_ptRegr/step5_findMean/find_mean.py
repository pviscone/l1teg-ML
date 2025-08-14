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
import numpy as np
import matplotlib.pyplot as plt

from common import signal_test,features_q, xgbmodel, conifermodel, cpp_cfg, quant
from file_utils import open_signal, quantize_features


#%%
xgb_model = xgb.XGBRegressor()
xgb_model.load_model(xgbmodel)


#!---------------------- HLS Model----------------------!#
hls_model = conifer.model.load_model(conifermodel, new_config=cpp_cfg)
print("Model converted")
hls_model.compile()
print("Model compiled")

#%%
df = open_signal(signal_test)
df = quantize_features(df, features_q, quant)

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