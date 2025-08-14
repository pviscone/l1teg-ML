#%%
# DO NOT CHANGE ORDER
import sys
sys.path.append("..")
sys.path = ["../utils/xgboost/python-package"] + sys.path
import xgboost

import numpy as np
from sklearn.model_selection import train_test_split
from bithub.quantizers import xilinx
from common import signal_train, bkg_train, eta_, genpt_, pt_, ptratio_dict, metric, quant, q_out, features_q, w, out_cut
from plot_utils import plot_xgb_loss, plot_xgb_importance, plot_results,plot_bkg
from xgb_loss import L1Metrics
from file_utils import open_signal, open_bkg, merge_signal_bkg, quantize_features, quantize_target

#?Open dfs
df_sig = open_signal(signal_train)
df_bkg = open_bkg(bkg_train, df_sig, flat_pt=True)
df = merge_signal_bkg(df_sig, df_bkg)
df = quantize_features(df, features_q, quant)

#? Split into train and test and mask the train
df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

train_mask = np.bitwise_and( (df_train["target"] < out_cut- 1./2**(q_out[0]-q_out[1])) , df_train[pt_].values >=2)
#train_mask = (df_train["target"] < out_cut- 1./2**(q_out[0]-q_out[1]))
df_train = df_train[train_mask]
df_train = quantize_target(df_train, q_out)


# %%
l1_metric={}
for k,dataframe in {"train":df_train, "test":df_test}.items():
    eval_set = [(dataframe[features_q], dataframe["target"])]

    #with abserr remember to pass sample weight
    l1_metric[k] = L1Metrics(dataframe[w].values, dataframe["label"].values)
    print("Showing metrics for", k)
    model = xgboost.XGBRegressor(
        max_depth=6,
        learning_rate=0.7,
        subsample=1.,
        colsample_bytree=1.0,
        alpha=0.,
        min_split_loss=5,
        min_child_weight=100,
        n_estimators=10,

        eval_metric=l1_metric[k].metrics,

        #?custom loss parameters
        #objective="reg:absoluteerror",
        objective="reg:l1loss",
        alphaL1=0.99,
        bkg_target=1.,
        cls_s=",".join(df_train["label"].values.astype(str).tolist()),
    )

    eval_result = {}
    model.fit(
        df_train[features_q],
        df_train["target"].values,
        eval_set=eval_set,
        sample_weight=df_train[w].values,
    )


plot_xgb_loss(l1_metric, savefolder=None)
plot_xgb_importance(model, features_q, savefolder=None)
print(len(model.get_booster().trees_to_dataframe()))
#%%
df_sig_test = df_test[df_test["label"].values==1]
df_sig_test["model_output"] = model.predict(df_sig_test[features_q])
df_sig_test["model_output_quantized"] = xilinx.convert(xilinx.ap_ufixed(q_out[0], q_out[1], q_mode = "AP_RND_CONV")(df_sig_test["model_output"].values), "double")
df_sig_test[ptratio_dict["Regressed"]] = df_sig_test["model_output_quantized"].values*df_sig_test[pt_].values/ df_sig_test[genpt_].values

plot_results(df_sig_test, ptratio_dict, genpt_, eta_, savefolder=f"plots/plots{metric}_q{quant}_out{q_out[0]}_{q_out[1]}")
plot_results(df_sig_test, ptratio_dict, genpt_, eta_, savefolder=f"plots/plots{metric}_q{quant}_out{q_out[0]}_{q_out[1]}", eta_bins=np.array([0,1.479]), verbose=False)
plot_bkg(model, df_test, features_q, pt_, what="all", savefolder=f"plots/plots{metric}_q{quant}_out{q_out[0]}_{q_out[1]}", verbose=True, bins = np.arange(4,38,3))

# %%
#model.save_model(f"../models/xgb_model_{metric}_q{quant}_out{q_out[0]}_{q_out[1]}.json")