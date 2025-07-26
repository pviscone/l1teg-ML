#%%
import sys
import os
sys.path.append("../utils")
sys.path.append("../../../utils/BitHub")

import numpy as np
import matplotlib.pyplot as plt
from file_utils import openAsDataframe
from compute_weights import cut_and_compute_weights
import xgboost
import pandas as pd
from bithub.quantizers import mp_xilinx

out_folder = "out"
collection = "TkEle"
eta_ = f"{collection}_caloEta"
genpt_ = f"{collection}_Gen_pt"
pt_ = f"{collection}_in_caloPt"
ptratio_dict = {"NoRegression": "TkEle_Gen_ptRatio",
                "Regressed": "TkEle_regressedPtRatio"}

metric = "L1"
quant = 10

if metric == "L1":
    loss = "reg:absoluteerror"
    eval_metric = "mae"
    w = "wTot"
elif metric == "L2":
    loss = "reg:squarederror"
    eval_metric = "rmse"
    w = "w2Tot"
else:
    raise ValueError("Unknown metric. Use 'L1' or 'L2'.")
#os.makedirs(f"{out_folder}", exist_ok=True)

if not os.path.exists("DoubleElectron_PU200.root"):
    raise ValueError("xrdcp root://eosuser.cern.ch//eos/user/p/pviscone/www/L1T/l1teg/EB_ptRegr/step0_ntuple/DoubleEle_PU200/zsnap/era151Xv0pre4_TkElePtRegr_dev_withScaled/base_2_ptRatioMultipleMatch05/DoubleElectron_PU200.root .")

os.makedirs(out_folder, exist_ok=True)

df = openAsDataframe("DoubleElectron_PU200.root", "TkEle")

features = [
    #'TkEle_in_caloStaWP',
    #'TkEle_in_caloTkAbsDeta',
    #'TkEle_in_caloLooseTkWP',
    #'TkEle_idScore',
    "hwCaloEta",
    'caloTkAbsDphi',
    'hwTkChi2RPhi',
    'caloPt',
    'caloRelIso',
    'caloSS',
    'tkPtFrac',
    'caloTkNMatch',
    'caloTkPtRatio',
]

df["hwCaloEta"] = np.clip(-1+df["TkEle_hwCaloEta"].abs()/2**8, -1, 1)
df["caloTkAbsDphi"] = np.clip(-1 + df["TkEle_in_caloTkAbsDphi"]/2**5, -1, 1)
df["hwTkChi2RPhi"] = np.clip(-1 + df["TkEle_in_hwTkChi2RPhi"]/2**3, -1, 1)
df["caloPt"] = np.clip(-1+(df["TkEle_in_caloPt"]-1)/2**6, -1, 1)
df["caloRelIso"] = np.clip(-1 + df["TkEle_in_caloRelIso"]*2**3, -1, 1)
df["caloSS"] = np.clip(-1 + df["TkEle_in_caloSS"]*2, -1, 1)
df["tkPtFrac"] = np.clip(-1 + df["TkEle_in_tkPtFrac"]/2**3, -1, 1)
df["caloTkNMatch"] = np.clip(-1 + df["TkEle_in_caloTkNMatch"]/2**2, -1, 1)
df["caloTkPtRatio"] = np.clip(-1 + df["TkEle_in_caloTkPtRatio"]/2**3, -1, 1)


df[features] = pd.DataFrame(
    mp_xilinx.mp_xilinx(df[features], f'ap_fixed<{quant}, 1, "AP_RND_CONV", "AP_SAT">', convert="double")
)

df = cut_and_compute_weights(df, genpt_, pt_, ptcut = 1)
#%%
df_train = df[features]
gen_train = df["TkEle_Gen_pt"]
ptratio_train = df["TkEle_Gen_ptRatio"]
eta_train = df[eta_]
pt_train = df[pt_]
dfw_train = df[["RESw", "BALw", "wTot","w2Tot"]]

# %%

model = xgboost.XGBRegressor(
    objective=loss,
    max_depth=7,
    learning_rate=0.7,
    subsample=1.,
    colsample_bytree=1.0,
    alpha=0.,
    lambd=0.00,
    min_split_loss=5,
    min_child_weight=120,
    n_estimators=15,
    eval_metric=eval_metric,
)
eval_set = [(df_train.values, gen_train.values/pt_train.values)]
eval_result = {}
model.fit(
    df_train.values,
    gen_train.values/pt_train.values,
    sample_weight=dfw_train[w].values,
    eval_set=eval_set,
)

# Plot training and validation loss
results = model.evals_result()
plt.figure(figsize=(8, 5))
plt.plot(results['validation_0'][eval_metric], label='Train')
#plt.plot(results['validation_1'][eval_metric], label='Validation')
plt.xlabel('Boosting Round')
plt.ylabel(f'{eval_metric}')
plt.title(f'Training and Validation {eval_metric}')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{out_folder}/loss_curve.png")
plt.savefig(f"{out_folder}/loss_curve.pdf")
plt.show()

# Plot feature importance
importances = model.feature_importances_
plt.figure(figsize=(10, 6))
plt.barh(features, importances)
plt.xlabel("Feature Importance")
plt.title("XGBoost Feature Importances")
plt.tight_layout()
plt.savefig(f"{out_folder}/feature_importance.png")
plt.savefig(f"{out_folder}/feature_importance.pdf")
plt.show()

# Save the model
model.save_model(f"{out_folder}/xgboost_model_{metric}_q{quant}.json")