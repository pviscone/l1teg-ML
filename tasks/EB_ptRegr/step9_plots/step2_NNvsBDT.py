#%%
import sys
import os
os.environ["KERAS_BACKEND"] = "jax"
sys.path.append("..")
import numpy as np
import xgboost as xgb

from common import signal_test, eta_, genpt_, pt_, features_q
from plot_utils import plot_results
from file_utils import open_signal
import keras

basename = __file__.split("/")[-1].split(".")[0]
savefolder = f"plots/{basename}"
os.makedirs(savefolder, exist_ok=True)


#?Open dfs
df = open_signal(signal_test)


#%%
ptratio_dict = {"NoRegression": "TkEle_Gen_ptRatio",
                "BDT": "BDT",
                "NN": "NN",
                }

bdt = xgb.XGBRegressor()
bdt.load_model("../models/xgboost_model_L1.json")
NN = keras.models.load_model("../models/NN_L1_model.keras")

df[ptratio_dict["BDT"]] = bdt.predict(df[features_q].values)*df[pt_].values/df[genpt_]
df[ptratio_dict["NN"]] = NN.predict(df[features_q].values)[:,0]*df[pt_].values/df[genpt_]

plot_results(df, ptratio_dict, genpt_, eta_, verbose=False, savefolder=savefolder)
plot_results(df, ptratio_dict, genpt_, eta_, verbose=False, savefolder=savefolder, eta_bins=np.array([0,1.479]))


