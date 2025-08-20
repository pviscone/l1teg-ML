#%%
import sys
import os
sys.path.append("..")
from common import signal_test, cpp_cfg, conifermodel, features_q, quant, init_pred, pt_, genpt_, conifer_signalonly, init_pred_signalonly

import numpy as np

import conifer
from file_utils import open_signal, quantize_features
from scaling_utils import derive_scaling

basename = __file__.split("/")[-1].split(".")[0]
savefolder = f"plots/{basename}"
os.makedirs(savefolder, exist_ok=True)

df = open_signal(signal_test)
df = df[(df["TkEle_hwQual"].values & 2) == 2 ]
df = quantize_features(df, features_q, quant)

model = conifer.model.load_model(conifermodel, new_config=cpp_cfg)
model.compile()

model_signalonly = conifer.model.load_model(conifer_signalonly, new_config=cpp_cfg)
model_signalonly.compile()

df["ptCorr"] = df[pt_].values * (init_pred + model.decision_function(df[features_q].values)[:,0])
df["ptCorrSignalOnly"] = df[pt_].values * (init_pred_signalonly + model_signalonly.decision_function(df[features_q].values)[:,0])

gen = df[genpt_].values
pt = df[pt_].values
ptCorr = df["ptCorr"].values
ptCorrSignalOnly = df["ptCorrSignalOnly"].values
#%%
pt_cuts = np.arange(15, 56, 10)
cuts, y95 = derive_scaling({"Non-Regressed": pt, "Regressed": ptCorr, "Regressed (Signal only)":ptCorrSignalOnly}, gen, pt_cuts=pt_cuts, verbose=True, savefolder=savefolder)
