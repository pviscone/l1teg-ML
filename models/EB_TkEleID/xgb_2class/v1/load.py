# %%
from utils.data import (
    generate_paths,
    load_df,
    normalize_weight,
    concatenate,
    df_to_DMatrix,
    take_max_score
)

from utils.plot.post_train import (
    plot_scores,
    plot_roc,
    plot_roc_bins,
)

from bithub.scalers import BitScaler

import numpy as np
import xgboost as xgb
import pandas as pd

from params import features, auxiliary, samples, tag, P0

# %%
#!------------------------------------- Load Dataframe -------------------------------------!#
paths = generate_paths(P0, tag, samples)
sig_train, sig_test, bkg_train, bkg_test = load_df(
    *paths, branches=features + auxiliary
)
sig_train, bkg_train = normalize_weight(
    sig_train, bkg_train, key="TkEle_weight", kind="entries"
)

df_train, df_test = concatenate(
    {"train": [sig_train, bkg_train], "test": [sig_test, bkg_test]}
)

# %%
#!------------------------------------ Load XGBoost Model -----------------------------------!#
quant = 9
pt_bins=(0, 5, 10, 20, 30, 50, 100)

model = xgb.Booster()
model.load_model(f"results/q_{quant}/model.json")

scaler = BitScaler()
scaler.load(f"results/q_{quant}/scaler.json")
dtrain, dtest = df_to_DMatrix(
    df_train,
    df_test,
    features=features,
    y="TkEle_label",
    bitscaler=scaler,
    ap_fixed=(quant,1,"AP_RND_CONV","AP_SAT") if quant != "float" else None,
    weight="TkEle_weight",
    class_weights="balanced",
)


#%%
#!------------------------------------ Evaluate -----------------------------------!#
raw_func = lambda x: np.log(x / (1 - x)) / 8
df_train["score"] = raw_func(model.predict(dtrain))
df_test["score"] = raw_func(model.predict(dtest))

sig_test["score"] = df_test["score"][df_test["TkEle_label"] == 1]
sig_train["score"] = df_train["score"][df_train["TkEle_label"] == 1]
bkg_test["score"] = df_test["score"][df_test["TkEle_label"] == 0]
bkg_train["score"] = df_train["score"][df_train["TkEle_label"] == 0]
#%%
#!------------------------------------ Best per cluster -----------------------------------!#


sig_train_best, sig_test_best = take_max_score(
    ["TkEle_ev_idx", "TkEle_GenEle_idx"], sig_train, sig_test
)
bkg_train_best, bkg_test_best = take_max_score(
    ["TkEle_ev_idx", "TkEle_CryClu_idx"], bkg_train, bkg_test
)

df_test_best = pd.concat([sig_test_best, bkg_test_best])
df_train_best = pd.concat([sig_train_best, bkg_train_best])


#%%
#!------------------------------------ Plot scores -----------------------------------!#
plot_scores(
    df_train,
    df_test[df_test["TkEle_CryClu_pt"] < df_train["TkEle_CryClu_pt"].max()],
    score="score",
    y="TkEle_label",
    bins=np.linspace(-1, 1, 30),
    log=True,
)

#!------------------------------------ Plot ROCs (only test) ----------------------------------!#
plot_roc(
    df_train,
    df_test[df_test["TkEle_CryClu_pt"] < df_train["TkEle_CryClu_pt"].max()],
    score="score",
    y="TkEle_label",
)
#!------------------------------------ Best per cluster ----------------------------------!#
plot_roc(
    df_train_best,
    df_test_best[
        df_test_best["TkEle_CryClu_pt"] < df_train_best["TkEle_CryClu_pt"].max()
    ],
    score="score",
    y="TkEle_label",
)

#!------------------------------------ ROC per pt ----------------------------------!#
_, aucs = plot_roc_bins(
    df_test_best,
    score="score",
    label="$p_T$",
    units="GeV",
    y="TkEle_label",
    var_name="TkEle_CryClu_pt",
    xlim=(-0.025, 0.5),
    var_bins=pt_bins,
)

plot_roc_bins(
    df_train_best,
    score="score",
    label="$p_T$",
    units="GeV",
    y="TkEle_label",
    var_name="TkEle_CryClu_pt",
    xlim=(-0.025, 0.5),
    var_bins=pt_bins,
)



