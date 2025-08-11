#%%
import sys
import os
sys.path.append("../utils")
os.environ["KERAS_BACKEND"] = "jax"
import numpy as np
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
df = cut_and_compute_weights(df, genpt_, pt_, ptcut=0)

df["TkEle_in_caloEta"] = df["TkEle_caloEta"].abs()-1

features = [
    "TkEle_in_caloEta",
    #'TkEle_in_caloStaWP',
    #'TkEle_in_caloTkAbsDeta',
    'TkEle_in_caloTkAbsDphi',
    'TkEle_in_tkChi2RPhi',
    #'TkEle_in_caloLooseTkWP',
    'TkEle_in_caloPt',
    #'TkEle_in_caloRelIso',
    'TkEle_in_caloSS',
    #'TkEle_in_tkPtFrac',
    #'TkEle_in_caloTkNMatch',
    'TkEle_in_caloTkPtRatio',
    #'TkEle_idScore',
]

df_train, df_test, gen_train, gen_test, ptratio_train, ptratio_test, eta_train, eta_test, dfw_train, dfw_test = train_test_split(df[features], df["TkEle_Gen_pt"], df["TkEle_Gen_ptRatio"], df[eta_], df[["RESw", "BALw", "wTot","w2Tot"]], test_size=0.2, random_state=42)

mask = gen_train.values/df_train[pt_].values<4
df_train = df_train[mask]
gen_train = gen_train[mask]
ptratio_train = ptratio_train[mask]
eta_train = eta_train[mask]
dfw_train = dfw_train[mask]

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
eval_set = [(df_train.values, gen_train.values/df_train[pt_].values), (df_test[features].values, gen_test.values/df_test[pt_].values)]
eval_result = {}
model.fit(
    df_train.values,
    gen_train.values/df_train[pt_].values,
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

#save model
model.save_model(f"plots{metric}/xgboost_model_{metric}.json")

#%%
from plot_utils import plot_ptratio_distributions_n_width, response_plot, resolution_plot  # noqa: E402
#evaluate on test set
def plot_results(model, plot_distributions=False, eta_bins=None):
    global ptratio_test, ptratio_dict, gen_test, genpt_, eta_test, eta_
    df_test[ptratio_dict["NoRegression"]] = ptratio_test
    df_test[genpt_] = gen_test
    df_test[eta_]=eta_test
    df_test[ptratio_dict["Regressed"]] = model.predict(df_test[features].values)*df_test[pt_].values/df_test[genpt_]



    eta_bins, centers, medians, perc5s, perc95s, perc16s, perc84s, residuals, variances, n , width = plot_ptratio_distributions_n_width(df_test,ptratio_dict,genpt_,eta_, genpt_bins=np.linspace(4,100,33), plots=plot_distributions, savefolder=f"plots{metric}", eta_bins=eta_bins)
    response_plot(ptratio_dict, eta_bins, centers, medians, perc5s, perc95s, perc16s, perc84s, residuals, variances, savefolder=f"plots{metric}")
    resolution_plot(ptratio_dict, eta_bins, centers, width, variances=variances, n=n, savefolder=f"plots{metric}")

plot_results(model, plot_distributions=False)
plot_results(model, plot_distributions=False, eta_bins=np.array([0,1.479]))
# %%
dump_file = open(f"plots{metric}/logger.log", "w")
model_df = model.get_booster().trees_to_dataframe()

print("MAX\n", file=dump_file)
print(model_df.groupby("Feature").max()["Split"], file=dump_file)

print("\nMIN\n", file=dump_file)
print(model_df.groupby("Feature").min()["Split"], file=dump_file)


print("\n", file=dump_file)
for i in range(len(features)):
    print(f"f{i}: {features[i]}", file=dump_file)

dump_file.close()

# %%
