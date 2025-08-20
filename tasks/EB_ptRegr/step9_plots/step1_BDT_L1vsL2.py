#%%
import sys
import os
sys.path.append("..")
import numpy as np
import xgboost as xgb

from common import signal_test, eta_, genpt_, pt_, features_q
from plot_utils import plot_results
from file_utils import open_signal

basename = __file__.split("/")[-1].split(".")[0]
savefolder = f"plots/{basename}"
os.makedirs(savefolder, exist_ok=True)


#?Open dfs
df = open_signal(signal_test)

#? Split into train and test and mask the train


modelL1 = xgb.XGBRegressor()
modelL1.load_model("../models/xgboost_model_L1.json")
modelL2 = xgb.XGBRegressor()
modelL2.load_model("../models/xgboost_model_L2.json")


#%%
ptratio_dict ={}
ptratio_dict["No regression"] = "TkEle_Gen_ptRatio"
ptratio_dict["Regressed (L1 Loss)"] = "TkEle_regressedPtRatio_L1"
ptratio_dict["Regressed (L2 Loss)"] = "TkEle_regressedPtRatio_L2"

df[ptratio_dict["Regressed (L1 Loss)"]] = modelL1.predict(df[features_q]) * df[pt_].values/ df[genpt_].values
df[ptratio_dict["Regressed (L2 Loss)"]] = modelL2.predict(df[features_q]) * df[pt_].values/ df[genpt_].values



plot_results(df, ptratio_dict, genpt_, eta_, verbose=False, savefolder=savefolder)
plot_results(df, ptratio_dict, genpt_, eta_, verbose=False, savefolder=savefolder, eta_bins=np.array([0,1.479]))

#%%
ptratio_dict ={}
ptratio_dict["No regression"] = "TkEle_Gen_ptRatio"
plot_results(df, ptratio_dict, genpt_, eta_, verbose=True, savefolder=savefolder, eta_bins=np.array([0,1.479]))
plot_results(df, ptratio_dict, genpt_, eta_, verbose=False, savefolder=savefolder, eta_bins=np.array([0,1.479]))