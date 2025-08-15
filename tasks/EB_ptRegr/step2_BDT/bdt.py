#%%
import sys
import os
sys.path.append("..")
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import xgboost

from common import signal_train, eta_, genpt_, pt_, ptratio_dict, features_q, get_loss_and_eval_metric
from plot_utils import plot_results
from file_utils import open_signal

#?Open dfs
df = open_signal(signal_train)

#? Split into train and test and mask the train
df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

train_mask = np.bitwise_and( df_train["target"] < 4 , df_train[pt_].values >=2)
#train_mask = (df_train["target"] < out_cut- 1./2**(q_out[0]-q_out[1]))
df_train = df_train[train_mask]

# %%
metric = "L2"
loss, eval_metric, w = get_loss_and_eval_metric(metric)

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
eval_set = [(df_train[features_q].values, df_train["target"].values), (df_test[features_q].values, df_test["target"].values)]
eval_result = {}
model.fit(
    df_train[features_q].values,
    df_train["target"].values,
    df_train[w].values,
    eval_set=eval_set,
)

os.makedirs(f"plots{metric}", exist_ok=True)
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
plt.barh(features_q, importances)
plt.xlabel("Feature Importance")
plt.title("XGBoost Feature Importances")
plt.tight_layout()
plt.savefig(f"plots{metric}/feature_importance.png")
plt.savefig(f"plots{metric}/feature_importance.pdf")
plt.show()

#%%

df_test[ptratio_dict["Regressed"]] = model.predict(df_test[features_q]) * df_test[pt_].values/ df_test[genpt_].values
plot_results(df_test, ptratio_dict, genpt_, eta_, verbose=False, savefolder=f"plots{metric}")
plot_results(df_test, ptratio_dict, genpt_, eta_, verbose=False, savefolder=f"plots{metric}", eta_bins=np.array([0,1.479]))
# %%
#save model
#model.save_model(f"../models/xgboost_model_{metric}.json")
