#%%
import sys
import os
sys.path.append("../utils")
os.environ["KERAS_BACKEND"] = "jax"
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from file_utils import openAsDataframe
from compute_weights import cut_and_compute_weights
import xgboost

collection = "TkEle"
eta_ = f"{collection}_caloEta"
genpt_ = f"{collection}_Gen_pt"
pt_ = f"{collection}_in_caloPt"
ptratio_dict = {"NoRegression": "TkEle_Gen_ptRatio",
                "Regressed": "TkEle_regressedPtRatio"}

metric = "L1"
ptcut=1

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
os.makedirs(f"plots{metric}", exist_ok=True)

if not os.path.exists("DoubleElectron_PU200.root"):
    raise ValueError("xrdcp root://eosuser.cern.ch//eos/user/p/pviscone/www/L1T/l1teg/EB_ptRegr/step0_ntuple/DoubleEle_PU200/zsnap/era151Xv0pre4_TkElePtRegr_dev_withScaled/base_2_ptRatioMultipleMatch05/DoubleElectron_PU200.root .")

df = openAsDataframe("DoubleElectron_PU200.root", "TkEle")
df["TkEle_scaled_caloAbsEta"] = df["TkEle_caloEta"].abs()-1

df_lessThan = df[df["TkEle_in_caloPt"]<ptcut]
df = cut_and_compute_weights(df, genpt_, pt_, ptcut=ptcut)


features = [
    "TkEle_scaled_caloAbsEta",
    #'TkEle_scaled_caloStaWP',
    #'TkEle_scaled_caloTkAbsDeta',
    'TkEle_scaled_caloTkAbsDphi',
    'TkEle_scaled_tkChi2RPhi',
    #'TkEle_scaled_caloLooseTkWP',
    'TkEle_scaled_caloPt',
    'TkEle_scaled_caloRelIso',
    'TkEle_scaled_caloSS',
    'TkEle_scaled_tkPtFrac',
    'TkEle_scaled_caloTkNMatch',
    'TkEle_scaled_caloTkPtRatio',
    #'TkEle_idScore',
]

df_train, df_test, gen_train, gen_test, ptratio_train, ptratio_test, eta_train, eta_test, pt_train, pt_test, dfw_train, dfw_test = train_test_split(df[features], df["TkEle_Gen_pt"], df["TkEle_Gen_ptRatio"], df[eta_], df[pt_], df[["RESw", "BALw", "wTot","w2Tot"]], test_size=0.2, random_state=42)


df_test = pd.concat([df_test, df_lessThan[features]])
gen_test = pd.concat([gen_test, df_lessThan["TkEle_Gen_pt"]]) 
ptratio_test = pd.concat([ptratio_test, df_lessThan["TkEle_Gen_ptRatio"]]) 
eta_test = pd.concat([eta_test, df_lessThan[eta_]]) 
pt_test = pd.concat([pt_test, df_lessThan[pt_]]) 

# %%

model = xgboost.XGBRegressor(
    objective=loss,
    max_depth=7,
    learning_rate=0.45,
    subsample=0.8,
    colsample_bytree=1.0,
    alpha=0.,
    lambd=150.00,
    min_split_loss=5,
    min_child_weight=80,
    n_estimators=15,
    eval_metric=eval_metric,
)
eval_set = [(df_train.values, gen_train.values/pt_train.values), (df_test[features].values, gen_test.values/pt_test.values)]
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
plt.plot(results['validation_1'][eval_metric], label='Validation')
plt.xlabel('Boosting Round')
plt.ylabel(f'{eval_metric}')
plt.title(f'Training and Validation {eval_metric}')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"plots{metric}/loss_curve.png")
plt.savefig(f"plots{metric}/loss_curve.pdf")
plt.show()

# Plot feature importance
importances = model.feature_importances_
plt.figure(figsize=(10, 6))
plt.barh(features, importances)
plt.xlabel("Feature Importance")
plt.title("XGBoost Feature Importances")
plt.tight_layout()
plt.savefig(f"plots{metric}/feature_importance.png")
plt.savefig(f"plots{metric}/feature_importance.pdf")
plt.show()

#%%
from plot_utils import plot_ptratio_distributions, response_plot  # noqa: E402
#evaluate on test set
def plot_results(model, plot_distributions=False):
    global ptratio_test, ptratio_dict, gen_test, genpt_, eta_test, eta_, pt_
    df_test[ptratio_dict["NoRegression"]] = ptratio_test
    df_test[genpt_] = gen_test
    df_test[eta_] = eta_test
    df_test[pt_] = pt_test
    df_test[ptratio_dict["Regressed"]] = model.predict(df_test[features].values)*df_test[pt_].values/df_test[genpt_]



    eta_bins, centers, medians, perc5s, perc95s, perc16s, perc84s, residuals, variances = plot_ptratio_distributions(df_test,ptratio_dict,genpt_,eta_, genpt_bins=np.linspace(1,100,34), plots=plot_distributions, savefolder=f"plots{metric}")
    response_plot(ptratio_dict, eta_bins, centers, medians, perc5s, perc95s, perc16s, perc84s, residuals, variances, savefolder=f"plots{metric}")

plot_results(model, plot_distributions=True)
# %%

