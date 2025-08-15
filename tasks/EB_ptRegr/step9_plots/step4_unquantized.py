#%%
# DO NOT CHANGE ORDER
import sys
import os
sys.path.append("..")
sys.path = ["../utils/xgboost/python-package"] + sys.path
import xgboost as xgb
from common import signal_test, eta_, genpt_, pt_, quant, q_out, features_q, xgbmodel, conifermodel, cpp_cfg, ptratio_dict

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
df = open_signal(signal_test)
df = quantize_features(df, features_q, quant)


#%%

df[ptratio_dict["Regressed"]] = xilinx.convert(xilinx.ap_ufixed(q_out[0], q_out[1], q_mode = "AP_RND_CONV")(
        xgb_model.predict(df[features_q])),
    "double")* df[pt_].values/ df[genpt_].values



plot_results(df, ptratio_dict, genpt_, eta_, savefolder=savefolder)
plot_results(df, ptratio_dict, genpt_, eta_, savefolder=savefolder, eta_bins=np.array([0,1.479]))
