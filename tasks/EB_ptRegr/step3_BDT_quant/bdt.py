#%%
import sys
import os
sys.path.append("../utils")
sys.path.append("../../../utils/BitHub")

import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from file_utils import openAsDataframe
from compute_weights import cut_and_compute_weights
import xgboost
import pandas as pd
from bithub.quantizers import mp_xilinx, xilinx

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
df_train, df_test, gen_train, gen_test, ptratio_train, ptratio_test, eta_train, eta_test, pt_train, pt_test, dfw_train, dfw_test = train_test_split(df[features], df["TkEle_Gen_pt"], df["TkEle_Gen_ptRatio"], df[eta_], df[pt_], df[["RESw", "BALw", "wTot","w2Tot"]], test_size=0.2, random_state=42)

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
os.makedirs(f"plots_q{quant}", exist_ok=True)
plt.savefig(f"plots_q{quant}/loss_curve.png")
plt.savefig(f"plots_q{quant}/loss_curve.pdf")
plt.show()

# Plot feature importance
importances = model.feature_importances_
plt.figure(figsize=(10, 6))
plt.barh(features, importances)
plt.xlabel("Feature Importance")
plt.title("XGBoost Feature Importances")
plt.tight_layout()
plt.savefig(f"plots_q{quant}/feature_importance.png")
plt.savefig(f"plots_q{quant}/feature_importance.pdf")
plt.show()

#%%
from plot_utils import plot_ptratio_distributions, response_plot  # noqa: E402
#evaluate on test set
def plot_results(model, plot_distributions=False, clip = 4, q_out=4):
    global ptratio_test, ptratio_dict, gen_test, genpt_, eta_test, eta_, pt_
    df_test[ptratio_dict["NoRegression"]] = ptratio_test
    df_test[genpt_] = gen_test
    df_test[eta_] = eta_test
    df_test[pt_] = pt_test
    df_test["model_output"] = np.clip(model.predict(df_test[features].values),0, clip)
    n_int_bits = int(np.ceil(np.log2(clip)))
    assert q_out > n_int_bits, f"q_out ({q_out}) must be greater than n_int_bits ({n_int_bits})"

    df_test["model_output_quantized"] = xilinx.convert(xilinx.ap_ufixed(q_out, n_int_bits, q_mode = "AP_RND_CONV")(df_test["model_output"].values), "double")

    df_test[ptratio_dict["Regressed"]] = df_test["model_output_quantized"].values*df_test[pt_].values/df_test[genpt_].values

    os.makedirs(f"plots/plots{metric}_q{quant}_out{q_out}_clip{clip}", exist_ok=True)


    eta_bins, centers, medians, perc5s, perc95s, perc16s, perc84s, residuals, variances = plot_ptratio_distributions(df_test,ptratio_dict,genpt_,eta_, genpt_bins=np.linspace(1,100,34), plots=plot_distributions, savefolder=f"plots/plots{metric}_q{quant}_out{q_out}_clip{clip}")
    response_plot(ptratio_dict, eta_bins, centers, medians, perc5s, perc95s, perc16s, perc84s, residuals, variances, savefolder=f"plots/plots{metric}_q{quant}_out{q_out}_clip{clip}")

for clip in [4]:
    for q_out in [9]:
        plot_results(model, q_out=q_out, clip=clip, plot_distributions=False)