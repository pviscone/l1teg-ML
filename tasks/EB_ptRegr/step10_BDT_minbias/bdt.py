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

eta_ = "TkEle_caloEta"
genpt_ = "TkEle_Gen_pt"
pt_ = "TkEle_in_caloPt"
ptratio_dict = {"NoRegression": "TkEle_Gen_ptRatio",
                "Regressed": "TkEle_regressedPtRatio"}
ptratio = ptratio_dict["NoRegression"]

metric = "L1"
quant = 10
q_out = (12,3)

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

features = [
    #'TkEle_in_caloStaWP',
    #'TkEle_in_caloTkAbsDeta',
    #'TkEle_in_caloLooseTkWP',
    'TkEle_idScore',
    "hwCaloEta",
    'caloTkAbsDphi',
    'hwTkChi2RPhi',
    'caloPt',
    #'caloRelIso',
    'caloSS',
    #'tkPtFrac',
    #'caloTkNMatch',
    'caloTkPtRatio',
]

#! Open and prepare double electron dataset
df = openAsDataframe("DoubleElectron_PU200.root", "TkEle")
def scale(df):
    df["hwCaloEta"] = np.clip(-1+df["TkEle_hwCaloEta"].abs()/2**8, -1, 1)
    df["caloTkAbsDphi"] = np.clip(-1 + df["TkEle_in_caloTkAbsDphi"]/2**5, -1, 1)
    df["hwTkChi2RPhi"] = np.clip(-1 + df["TkEle_in_hwTkChi2RPhi"]/2**3, -1, 1)
    df["caloPt"] = np.clip(-1+(df["TkEle_in_caloPt"]-1)/2**6, -1, 1)
    df["caloRelIso"] = np.clip(-1 + df["TkEle_in_caloRelIso"]*2**3, -1, 1)
    df["caloSS"] = np.clip(-1 + df["TkEle_in_caloSS"]*2, -1, 1)
    df["tkPtFrac"] = np.clip(-1 + df["TkEle_in_tkPtFrac"]/2**3, -1, 1)
    df["caloTkNMatch"] = np.clip(-1 + df["TkEle_in_caloTkNMatch"]/2**2, -1, 1)
    df["caloTkPtRatio"] = np.clip(-1 + df["TkEle_in_caloTkPtRatio"]/2**3, -1, 1)
    return df

df = scale(df)
df[features] = pd.DataFrame(
    mp_xilinx.mp_xilinx(df[features], f'ap_fixed<{quant}, 1, "AP_RND_CONV", "AP_SAT">', convert="double")
)
df = cut_and_compute_weights(df, genpt_, pt_, ptcut = 0)
df["label"] = 1

#%%
#!Load minbias
import uproot
minbias = uproot.open("minbias.root")["Events"].arrays(library="pd")
minbias = scale(minbias)
minbias[features] = pd.DataFrame(
    mp_xilinx.mp_xilinx(minbias[features], f'ap_fixed<{quant}, 1, "AP_RND_CONV", "AP_SAT">', convert="double")
)
minbias["VARw"] = 1.
minbias["RESw"] = 1.
minbias["BALw"] = 1.
minbias["BALw"] = 1.
minbias["wTot"] = 1.
minbias["w2Tot"] = 1.
minbias["label"] = 0
minbias["target"] = 1.


#%%
df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
minbias_train, minbias_test = train_test_split(minbias, test_size=0.2, random_state=42)
df_train["target"] = xilinx.convert(xilinx.ap_fixed(q_out[0], q_out[1], "AP_RND_CONV", "AP_SAT")(df_train[genpt_].values/df_train[pt_].values), "double")
df_test["target"] = xilinx.convert(xilinx.ap_fixed(q_out[0], q_out[1], "AP_RND_CONV", "AP_SAT")(df_test[genpt_].values/df_test[pt_].values), "double")

#train_mask = (pt_ratio_q_train < 2.- 1./2**(q_out[0]-q_out[1])) & (pt_ratio_q_train > 0.5)
train_mask = (df_train["target"] < 4.- 1./2**(q_out[0]-q_out[1]))
df_train = df_train[train_mask]


# %%
model = xgboost.XGBRegressor(
    objective=loss,
    max_depth=6,
    learning_rate=0.7,
    subsample=1.,
    colsample_bytree=1.0,
    alpha=0.,
    lambd=0.00,
    min_split_loss=5,
    min_child_weight=100,
    n_estimators=12,
    eval_metric=eval_metric,

)

train = pd.concat([df_train, minbias_train])
test = pd.concat([df_test, minbias_test])
train = train[train[pt_]>3]

eval_set = [(train[features], train["target"]), (test[features], test["target"])]
eval_result = {}
model.fit(
    train[features],
    train["target"],
    sample_weight=train[w].values,
    eval_set=eval_set,
)
model.save_model("model.json")
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
os.makedirs(f"plots/plots_q{quant}", exist_ok=True)
plt.savefig(f"plots/plots_q{quant}/loss_curve.png")
plt.savefig(f"plots/plots_q{quant}/loss_curve.pdf")
plt.show()

# Plot feature importance
importances = model.feature_importances_
plt.figure(figsize=(10, 6))
plt.barh(features, importances)
plt.xlabel("Feature Importance")
plt.title("XGBoost Feature Importances")
plt.tight_layout()
plt.savefig(f"plots/plots_q{quant}/feature_importance.png")
plt.savefig(f"plots/plots_q{quant}/feature_importance.pdf")
plt.show()

#%%
from plot_utils import plot_ptratio_distributions_n_width, response_plot, resolution_plot  # noqa: E402
#evaluate on test set
def plot_results(model, plot_distributions=False, q_out=(11,2), eta_bins=None):
    df_test[ptratio_dict["NoRegression"]] = 1/df_test["target"]
    df_test["model_output"] = model.predict(df_test[features])

    df_test["model_output_quantized"] = xilinx.convert(xilinx.ap_ufixed(q_out[0], q_out[1], q_mode = "AP_RND_CONV")(df_test["model_output"].values), "double")

    df_test[ptratio_dict["Regressed"]] = df_test["model_output_quantized"].values*df_test[pt_].values/df_test[genpt_].values

    os.makedirs(f"plots/plots{metric}_q{quant}_out{q_out[0]}_{q_out[1]}", exist_ok=True)


    eta_bins, centers, medians, perc5s, perc95s, perc16s, perc84s, residuals, variances,n, width = plot_ptratio_distributions_n_width(df_test,ptratio_dict,genpt_,eta_, genpt_bins=np.linspace(1,100,34), plots=plot_distributions, eta_bins=eta_bins, savefolder=f"plots/plots{metric}_q{quant}_out{q_out[0]}_{q_out[1]}")
    response_plot(ptratio_dict, eta_bins, centers, medians, perc5s, perc95s, perc16s, perc84s, residuals, variances, savefolder=f"plots/plots{metric}_q{quant}_out{q_out[0]}_{q_out[1]}", verbose=False)
    resolution_plot(ptratio_dict, eta_bins, centers, width, variances=variances, n=n, savefolder=f"plots/plots{metric}_q{quant}_out{q_out[0]}_{q_out[1]}")

plot_results(model, q_out=q_out, plot_distributions=False)
plot_results(model, q_out=q_out, eta_bins=np.array([0,1.479]), plot_distributions=False)
# %%

#! plot 2d
plt.figure()
plt.hist2d(df_test[pt_], model.predict(df_test[features]), bins=(100, 100), range=((0, 100), (0, 4)), cmap='viridis', cmin=1e-99, density=True)
plt.xlabel("pT [GeV]")
plt.ylabel("Out")
plt.colorbar(label='Density')
plt.title("Signal")


#! plot 2d
plt.figure()
plt.hist2d(minbias[pt_], model.predict(minbias[features]), bins=(100, 100), range=((0, 100), (0, 4)), cmap='viridis', cmin=1e-99, density=True)
plt.xlabel("pT [GeV]")
plt.ylabel("Out")
plt.colorbar(label='Density')
plt.title("Minbias")

#%%
#! Online rate
import hist
import sys
import os
sys.path.append("../../../cmgrdf-cli/cmgrdf_cli/plots")

from plotters import TRate
rate = TRate(ylim=(1, 4e4), xlim=(-1,80), xlabel = "Online $p_{T}$ [GeV]", cmstext="Phase-2 Simulation Preliminary", lumitext="PU 200")

def fill(array, frac):
    axis = hist.axis.Regular(100, 0, 100)
    h = hist.Hist(axis, storage=hist.storage.Weight())
    h.fill(array)
    h = h/h.integrate(0).value
    h = h * 31000 * frac
    return h

h_pt_tight = fill(minbias_test[pt_],0.25483870967741934)
h_pt_corr_tight = fill(minbias_test[pt_]*model.predict(minbias_test[features]),0.25483870967741934)



rate.add(h_pt_corr_tight, label="Regressed (Tight)")
rate.add(h_pt_tight, label="No Regression (Tight)")
rate.ax.axhline(18, color="gray", linestyle="--", alpha=0.5)
rate.save("plots/online_rate.pdf")
rate.save("plots/online_rate.png")
# %%

#!Scaling
def scaling(x):
    return 1.09*x+2.82

def scaling_corr(x):
    return 1.06*x+2.36

rate = TRate(ylim=(1, 4e4), xlim=(-1,80), xlabel = "Offline $p_{T}$ [GeV]", cmstext="Phase-2 Simulation Preliminary", lumitext="PU 200")

def fill(array, frac):
    axis = hist.axis.Regular(100, 0, 100)
    h = hist.Hist(axis, storage=hist.storage.Weight())
    h.fill(array)
    h = h/h.integrate(0).value
    h = h * 31000 * frac
    return h

h_pt_tight = fill(scaling(minbias_test[pt_]),0.25483870967741934)
h_pt_corr_tight = fill(scaling_corr(minbias_test[pt_]*model.predict(minbias_test[features])),0.25483870967741934)



rate.add(h_pt_corr_tight, label="Regressed (Tight)")
rate.add(h_pt_tight, label="No Regression (Tight)")
rate.ax.axhline(18, color="gray", linestyle="--", alpha=0.5)
rate.save("plots/offline_rate.pdf")
rate.save("plots/offline_rate.png")
# %%
