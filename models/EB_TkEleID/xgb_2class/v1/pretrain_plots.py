# %%
from utils.data import (
    generate_paths,
    load_df,
    normalize_weight,
)
from utils.plot.features import profile, plot_input_features
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
# %%
#!------------------------------------ Plot Pre-training -----------------------------------!#

profile(sig_train, bkg_train, features=features, save="results/plots/profile_{name}")
plot_input_features(
    sig_train,
    bkg_train,
    weight="TkEle_weight",
    features=features,
    save="results/plots/input_features_reweight",
)
plot_input_features(
    sig_test,
    bkg_test,
    weight="TkEle_weight",
    features=features,
    save="results/plots/input_features",
)
