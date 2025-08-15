#%%
# DO NOT CHANGE ORDER
import sys
import os
sys.path.append("..")
sys.path.append("../../../utils/conifer")
sys.path = ["../utils/xgboost/python-package"] + sys.path
import conifer
import xgboost as xgb
from common import signal_test, eta_, genpt_, pt_, quant, q_out, features_q, xgbmodel, conifermodel, init_pred, cpp_cfg

import numpy as np
from bithub.quantizers import xilinx
from plot_utils import plot_results
from file_utils import open_signal, quantize_features

basename = __file__.split("/")[-1].split(".")[0]
savefolder = f"plots/{basename}"
os.makedirs(savefolder, exist_ok=True)

xgb_model = xgb.XGBRegressor()
xgb_model.load_model(xgbmodel)

#%%
#!----------------------Convert Model----------------------!#
hls_model = conifer.model.load_model(conifermodel, new_config=cpp_cfg)
print("Model converted")
hls_model.compile()
print("Model compiled")


#%%
df = open_signal(signal_test)
df = quantize_features(df, features_q, quant)


#%%

ptratio_dict={"BDT" : "TkEle_regressedPtRatioBDT","Quantized BDT" : "TkEle_regressedPtRatioBDTQuantized", }
df[ptratio_dict["BDT"]] = xilinx.convert(xilinx.ap_ufixed(q_out[0], q_out[1], q_mode = "AP_RND_CONV")(
        xgb_model.predict(df[features_q])),
    "double")* df[pt_].values/ df[genpt_].values

df[ptratio_dict["Quantized BDT"]] = xilinx.convert(
    xilinx.ap_ufixed(q_out[0], q_out[1], q_mode = "AP_RND_CONV")(
        (init_pred+hls_model.decision_function(df[features_q].values)[:,0])), "double"
) * df[pt_].values / df[genpt_].values


plot_results(df, ptratio_dict, genpt_, eta_, savefolder=savefolder)
plot_results(df, ptratio_dict, genpt_, eta_, savefolder=savefolder, eta_bins=np.array([0,1.479]))
