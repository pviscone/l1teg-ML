# %%
from utils.data import (
    generate_paths,
    load_df,
    normalize_weight,
    concatenate,
    df_to_DMatrix
)
from utils.plot.features import (
    profile,
    plot_input_features
)

from utils.plot.post_train import (
    plot_loss,
    plot_importance,
    plot_scores,
    plot_roc
)

import numpy as np
import xgboost as xgb

#!------------------------------------- Configuration -------------------------------------!#

tag = "140Xv0B9"
P0 = f"/eos/user/p/pviscone/www/L1T/l1teg/EB_pretrain/v0/zsnap/era{tag}/reweight"

features = [
    "CryClu_pt",
    "CryClu_showerShape",
    "CryClu_relIso",
    "CryClu_standaloneWP",
    "CryClu_looseL1TkMatchWP",
    "Tk_chi2RPhi",
    "Tk_ptFrac",
    "PtRatio",
    "nTkMatch",
    "absdeta",
    "absdphi",]

auxiliary = ["GenEle_pt", "label", "weight", "sumTkPt", "CryClu_idx"]

samples=[
        "DoubleElectron_PU200_train",
        "DoubleElectron_PU200_test",
        "MinBias_train",
        "MinBias_test",
    ]


features = [f"TkEle_{f}" for f in features]
auxiliary = [f"TkEle_{f}" for f in auxiliary]
#%%
#!------------------------------------- Load Dataframe -------------------------------------!#
paths = generate_paths(P0, tag, samples)
sig_train, sig_test, bkg_train, bkg_test = load_df(*paths, branches=features+auxiliary)
sig_train, bkg_train = normalize_weight(
    sig_train, bkg_train, key="TkEle_weight", kind="entries"
)
# %%
#!------------------------------------ Plot Pre-training -----------------------------------!#

profile(sig_train, bkg_train, features = features, save="results/plots/profile_{name}")
plot_input_features(sig_train, bkg_train, weight="TkEle_weight", features=features, save = "results/plots/input_features_reweight")
plot_input_features(sig_test, bkg_test, weight="TkEle_weight", features=features, save = "results/plots/input_features")

# %%
#!------------------------------------ Train XGBoost Model -----------------------------------!#
params = {
    "tree_method": "hist",
    "max_depth": 12,
    "learning_rate": 0.45,
    "lambda": 1500,
    "alpha": 1500,
    # "colsample_bytree":0.8,
    "subsample": 0.8,
    # "gamma":5,
    "min_split_loss": 5,
    "min_child_weight": 80,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
}
num_round = 15

df_train, df_test = concatenate({"train":[sig_train, bkg_train], "test":[sig_test, bkg_test]})

dtrain, dtest = df_to_DMatrix(
    df_train, df_test, features=features, label="TkEle_label", weight="TkEle_weight", class_weights="balanced"
)
evallist = [(dtrain, "train"), (dtest, "eval")]
eval_result = {}
model = xgb.train(params, dtrain, num_round, evallist, evals_result=eval_result)

raw_func = lambda x : np.log(x/(1-x))/8
df_train["score"] = raw_func(model.predict(dtrain))
df_test["score"] = raw_func(model.predict(dtest))

# %%
#!------------------------------------ Plot Loss and scores -----------------------------------!#
plot_loss(eval_result, save="results/plots/loss")
plot_importance(model, save="results/plots/importance")
plot_scores(df_train, df_test, score="score", label="TkEle_label", save="results/plots/scores", bins = np.linspace(-1,1,30), log = True)

# %%
#!------------------------------------ Plot ROCs (only test) ----------------------------------!#

plot_roc(df_train, df_test, score="score", label="TkEle_label", save="results/plots/roc")
# %%
#!------------------------------------ Best per cluster ----------------------------------!#
df_train_best = df_train.groupby(["TkEle_ev_idx", "TkEle_CryClu_idx", "TkEle_label"]).max("TkEle_score").reset_index()
df_test_best = df_test.groupby(["TkEle_ev_idx", "TkEle_CryClu_idx", "TkEle_label"]).max("TkEle_score").reset_index()
plot_roc(df_train_best, df_test_best, score="score", label="TkEle_label", save="results/plots/roc_bestTkEle")

# %%
#!------------------------------------ ROC per pt ----------------------------------!#
