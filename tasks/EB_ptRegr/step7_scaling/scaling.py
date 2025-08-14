#%%
import sys
sys.path.append("..")
from common import signal_test, cpp_cfg, conifermodel, features_q, quant, init_pred, pt_, genpt_

import numpy as np

import conifer
from file_utils import open_signal, quantize_features
from scaling_utils import derive_scaling



df = open_signal(signal_test)
df = df[(df["TkEle_hwQual"].values & 2) == 2 ]
df = quantize_features(df, features_q, quant)

model = conifer.model.load_model(conifermodel, new_config=cpp_cfg)
model.compile()

df["ptCorr"] = df[pt_].values * (init_pred + model.decision_function(df[features_q].values)[:,0])

gen = df[genpt_].values
pt = df[pt_].values
ptCorr = df["ptCorr"].values

#%%
pt_cuts = np.arange(15, 56, 10)
cuts, y95 = derive_scaling({"Non-Regressed": pt, "Regressed": ptCorr}, gen, pt_cuts=pt_cuts, verbose=False, savefolder="plots")
# %%
