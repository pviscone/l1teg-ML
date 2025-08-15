#%%
import sys
sys.path.append("..")
sys.path = ["../utils/xgboost/python-package"] + sys.path
import xgboost
from common import features_q, xgbmodel
import os
import matplotlib.pyplot as plt
import mplhep as hep
from labels import feature_labels
hep.style.use("CMS")

basename = __file__.split("/")[-1].split(".")[0]
savefolder = f"plots/{basename}"
os.makedirs(savefolder, exist_ok=True)

def plot_importance(model, features, log=False, xlim=None, save=None):
    importances = model.feature_importances_
    plt.figure(figsize=(10, 6))
    #argsort importances
    sorted_indices = importances.argsort()
    features = [feature_labels[features[i]].split("[")[0] for i in sorted_indices]
    importances = importances[sorted_indices]
    plt.barh(features, importances)
    plt.xlabel("Feature Importance")
    if log:
        plt.xscale("log")
    if xlim is not None:
        plt.xlim(xlim)
    hep.cms.text("Phase-2 Simulation Preliminary")
    plt.tight_layout()
    if save:
        plt.savefig(f"{savefolder}/feature_importance_{save}.png")
        plt.savefig(f"{savefolder}/feature_importance_{save}.pdf")
    plt.show()

#%%
model = xgboost.XGBRegressor()
model.load_model('../models/xgboost_model_L1.json')
plot_importance(model, features_q, save="signal")
#%%
model = xgboost.XGBRegressor()
model.load_model(xgbmodel)
plot_importance(model, features_q, log=True, xlim=(0.01,1), save="all")
# %%
