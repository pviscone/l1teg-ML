#%%
import sys
sys.path.append("../utils")
sys.path.append("..")
sys.path.append("../../../utils/BitHub")
sys.path = ["../utils/xgboost/python-package"] + sys.path

import numpy as np
from sklearn.model_selection import train_test_split
from file_utils import openAsDataframe
from compute_weights import cut_and_compute_weights, flat_w
import xgboost
import pandas as pd
from bithub.quantizers import mp_xilinx, xilinx
from common import signal_train, bkg_train, eta_, genpt_, pt_, ptratio_dict, metric, quant, q_out, features_q, scale, w, out_cut
from plot_utils import plot_xgb_loss, plot_xgb_importance, plot_results,plot_bkg
from xgb_loss import L1Metrics

#? Open signal
df_sig = openAsDataframe(signal_train, "TkEle")
df_sig = df_sig[df_sig[pt_]>0]
df_sig = scale(df_sig)
df_sig["label"] = 1.
df_sig = cut_and_compute_weights(df_sig, genpt_, pt_, ptcut = 0)

#? Open background
df_bkg = openAsDataframe(bkg_train, "TkEle")
df_bkg = df_bkg[df_bkg[pt_]>0]
df_bkg = scale(df_bkg)
df_bkg["TkEle_Gen_pt"]= df_bkg[pt_].values
df_bkg["TkEle_Gen_ptRatio"] = 1.
df_bkg["label"] = 0.
#%%
flat_pt = True
if flat_pt:
    df_bkg["BALw"] = flat_w(df_bkg[pt_].values, df_sig[pt_].values, weight = df_sig["BALw"].values)
else:
    df_bkg["BALw"] = np.ones(len(df_bkg)) * len(df_sig) / len(df_bkg)

df_bkg[w]=df_bkg["BALw"]

#? Concatenate signal and background
ks = list(set(df_sig.keys()).intersection(set(df_bkg.keys())))
df = pd.concat([df_sig[ks], df_bkg[ks]])

#? Apply quantization
df_quant = pd.DataFrame(
    mp_xilinx.mp_xilinx({k:df[k].values for k in features_q}, f'ap_fixed<{quant}, 1, "AP_RND_CONV", "AP_SAT">', convert="double")
)
for k in features_q:
    df[k] = df_quant[k].values
df["target"] = xilinx.convert(xilinx.ap_fixed(q_out[0], q_out[1], "AP_RND_CONV", "AP_SAT")(1/df["TkEle_Gen_ptRatio"].values), "double")

#? Split into train and test and mask the train
df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

#train_mask = np.bitwise_and( (df_train["target"] < out_cut- 1./2**(q_out[0]-q_out[1])) , df_train[pt_].values > 4)
train_mask = (df_train["target"] < out_cut- 1./2**(q_out[0]-q_out[1]))
df_train = df_train[train_mask]


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
        min_split_loss=10,
        min_child_weight=100,
        n_estimators=10,

        eval_metric=l1_metric[k].metrics,

        #?custom loss parameters
        #objective="reg:absoluteerror",
        objective="reg:l1loss",
        alphaL1=0.98,
        bkg_target=0.95,
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