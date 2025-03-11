# %%
from utils.data import (
    generate_paths,
    load_df,
    normalize_weight,
    concatenate,
    df_to_DMatrix,
)

from utils.plot.post_train import (
    plot_loss,
    plot_importance,
    plot_scores,
    plot_roc,
    plot_roc_bins,
    plot_quant_aucs,
)

from bithub.scalers import BitScaler

from bayes_opt import BayesianOptimization
import numpy as np
import xgboost as xgb

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
#!------------------------------------  scaling -----------------------------------!#

df_train, df_test = concatenate(
    {"train": [sig_train, bkg_train], "test": [sig_test, bkg_test]}
)


# scaler = None

# %%
#!------------------------------------ Train XGBoost Model -----------------------------------!#
# what = train, optimize
what = "optimize"

quantizations = [6, 8, 9, 10, 12, "float"]
quant_aucs = {}
quant_models = {}
quant_params = {}
scaler_quant = {}
pt_bins = (0, 5, 10, 20, 30, 50, 100)
for quant in quantizations:
    scaler = BitScaler()
    scaler.fit(
        df_train,
        columns=features,
        target=(-1 , 1 ),
        saturate={"TkEle_PtRatio": (0, 16),
                  "TkEle_Tk_chi2RPhi": (0, 64),
                  "TkEle_Tk_ptFrac": (0, 1),
                  "TkEle_CryClu_relIso": (0, 16),
                  "TkEle_CryClu_pt": (0, 64),
                  "TkEle_CryClu_showerShape": (0, 1),
                  },
        precision = quant-1 if quant != "float" else None
    )

    q = "float" if quant == "float" else (quant, 1, "AP_RND_CONV", "AP_SAT")
    print(f"Quantization: {q}")
    dtrain, dtest, dtest_cut = df_to_DMatrix(
        df_train,
        df_test,
        df_test[df_test["TkEle_CryClu_pt"] < df_train["TkEle_CryClu_pt"].max()],
        features=features,
        y="TkEle_label",
        bitscaler=scaler,
        ap_fixed=q if q != "float" else None,
        weight="TkEle_weight",
        class_weights="balanced",
    )

    model = None
    eval_result = None

    def train(
        max_depth=12,
        learning_rate=0.45,
        subsample=0.8,
        colsample_bytree=1.0,
        alpha=1500,
        lambd=1500,
        min_split_loss=5,
        min_child_weight=80,
        num_round=15,
    ):
        global model, eval_result, dtrain, dtest_cut, params, _num_round
        _num_round = num_round
        params = {
            "tree_method": "hist",
            "max_depth": int(max_depth),
            "learning_rate": learning_rate,
            "lambda": lambd,
            "alpha": alpha,
            "colsample_bytree": colsample_bytree,
            "subsample": subsample,
            # "gamma":5,
            "min_split_loss": min_split_loss,
            "min_child_weight": min_child_weight,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
        }
        num_round = 15
        evallist = [(dtrain, "train"), (dtest_cut, "eval")]
        eval_result = {}
        model = xgb.train(
            params,
            dtrain,
            num_round,
            evallist,
            evals_result=eval_result,
            early_stopping_rounds=2,
        )
        print(params)
        return -eval_result["eval"][params["eval_metric"]][-1]

    if what == "optimize":
        pbounds = {
            "max_depth": (10, 13),
            "learning_rate": (0.36, 0.6),
            "subsample": (0.7, 1.0),
            "colsample_bytree": (0.8, 1.0),
            "alpha": (100, 300),
            "lambd": (300, 500),
            "min_split_loss": (10, 15),
            "min_child_weight": (60, 100),
        }
        optimizer = BayesianOptimization(f=train, pbounds=pbounds, random_state=666)
        optimizer.maximize(init_points=10, n_iter=10)
        train(**optimizer.max["params"])
        print(optimizer.max["params"])
        train(**optimizer.max["params"])

    elif what == "train":
        fixed_params = {
            "alpha": 115.86842537080673,
            "colsample_bytree": 0.8973561043086346,
            "lambd": 330.73478040000134,
            "learning_rate": 0.558831631980591,
            "max_depth": 10,
            "min_child_weight": 70.81635815001403,
            "min_split_loss": 12.805172105675538,
            "subsample": 0.9707141169676181,
        }
        train(**fixed_params)

    plot_loss(eval_result, save=f"results/plots/q_{quant}/loss")

    #!------------------------------------ Evaluate -----------------------------------!#
    raw_func = lambda x: np.log(x / (1 - x)) / 8
    df_train["score"] = raw_func(model.predict(dtrain))
    df_test["score"] = raw_func(model.predict(dtest))
    df_train_best = (
        df_train.groupby(["TkEle_ev_idx", "TkEle_CryClu_idx", "TkEle_label"])
        .max("TkEle_score")
        .reset_index()
    )
    df_test_best = (
        df_test.groupby(["TkEle_ev_idx", "TkEle_CryClu_idx", "TkEle_label"])
        .max("TkEle_score")
        .reset_index()
    )

    #!------------------------------------ Plot Loss and scores -----------------------------------!#
    plot_importance(model, save=f"results/plots/q_{quant}/importance")
    plot_scores(
        df_train,
        df_test[df_test["TkEle_CryClu_pt"] < df_train["TkEle_CryClu_pt"].max()],
        score="score",
        y="TkEle_label",
        save=f"results/plots/q_{quant}/scores",
        bins=np.linspace(-1, 1, 30),
        log=True,
    )

    #!------------------------------------ Plot ROCs (only test) ----------------------------------!#
    plot_roc(
        df_train,
        df_test[df_test["TkEle_CryClu_pt"] < df_train["TkEle_CryClu_pt"].max()],
        score="score",
        y="TkEle_label",
        save=f"results/plots/q_{quant}/roc",
    )
    #!------------------------------------ Best per cluster ----------------------------------!#
    plot_roc(
        df_train_best,
        df_test_best[
            df_test_best["TkEle_CryClu_pt"] < df_train_best["TkEle_CryClu_pt"].max()
        ],
        score="score",
        y="TkEle_label",
        save=f"results/plots/q_{quant}/roc_bestTkEle",
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
        save=f"results/plots/q_{quant}/roc_pt_bestTkEle_test",
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
        save=f"results/plots/q_{quant}/roc_pt_bestTkEle_train",
    )
    quant_aucs[f"{quant}"] = aucs
    quant_models[quant] = model
    quant_params[quant] = params
    scaler_quant[quant] = scaler

# %%
plot_quant_aucs(quant_aucs, pt_bins, save="results/plots/quant_aucs")


# %%
##SAVE
# model
# scaler
# quantizations
# parameters
def save(quant, path):
    import json
    import os
    model = quant_models[quant]
    param = quant_params[quant]
    scaler = scaler_quant[quant]
    os.makedirs(path, exist_ok=True)
    model.save_model(f"{path}/model.json")
    scaler.save(f"{path}/scaler.json")
    param["quantizations"] = quant
    param["num_round"] = _num_round
    with open(f"{path}/parameters.json", "w") as f:
        f.write(json.dumps(param, indent=4))


    with open(f"{path}/report.txt", "w") as f:
        f.write("--- Scaler ---\n")
        f.write("\n inf + (x - min) >> bit_shift\n\n")
        f.write(str(scaler))
        f.write("\n\n--- Parameters ---\n")
        f.write(str(param))


def save_all(path):
    for quant in quantizations:
        save(quant, f"{path}/q_{quant}")


save_all("results")

# %%
