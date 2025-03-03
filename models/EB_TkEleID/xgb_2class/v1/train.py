# %%
from utils.data import (
    generate_paths,
    load_df,
    normalize_weight,
    compute_class_weights,
    select_columns,
)
from utils.plot.features import profile, plot_input_features

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

auxiliary = ["ev_idx", "GenEle_pt", "label", "weight", "sumTkPt"]

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
sig_train, sig_test, bkg_train, bkg_test = load_df(*paths)
sig_train, bkg_train = normalize_weight(
    sig_train, bkg_train, key="TkEle_weight", kind="entries"
)
sig_w, bkg_w = compute_class_weights(sig_train, bkg_train)
sig_train_skimmed, sig_test_skimmed, bkg_train_skimmed, bkg_test_skimmed = select_columns(
    sig_train, sig_test, bkg_train, bkg_test, columns=features
)
# %%
#!------------------------------------ Plot Pre-training -----------------------------------!#

profile(sig_train_skimmed, bkg_train_skimmed, save="figures/profile_{name}")
plot_input_features(sig_train, bkg_train, weight="TkEle_weight", features=features, save = "figures/input_features_reweight")
plot_input_features(sig_test, bkg_test, weight="TkEle_weight", features=features, save = "figures/input_features")

# %%
