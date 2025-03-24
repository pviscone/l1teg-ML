#%%
import os
import conifer

import numpy as np
import xgboost as xgb

from utils.data import (
    generate_paths,
    load_df,
    concatenate,
    df_to_DMatrix
)

from bithub.scalers import BitScaler

from params import features, auxiliary, tag, P0

import matplotlib.pyplot as plt
import mplhep as hep
hep.set_style("CMS")


#%%
#!----------------------Load Data----------------------!#
def get_dmatrix(precision = None, q_scaler="float"):
    paths = generate_paths(P0, tag, [
        "DoubleElectron_PU200_test",
        "MinBias_test",
    ])
    sig_test, bkg_test = load_df(
        *paths, branches=features + auxiliary
    )
    bkg_test=bkg_test[:len(sig_test)]
    df_test = concatenate(
        {"test": [sig_test, bkg_test]}
    )

    scaler = BitScaler()
    scaler.load(f"results/q_{q_scaler}/scaler.json")


    dmatrix = df_to_DMatrix(
        df_test,
        features=features,
        y="TkEle_label",
        bitscaler=scaler,
        ap_fixed=(precision,1,"AP_RND_CONV", "AP_SAT") if precision else None
    )
    return df_test, dmatrix



#%%
def convert(backend, precision, build = False, predict = False):
    df_test, dmatrix = get_dmatrix(q_scaler=precision)

    model = f"results/q_{precision}/model.json"
    #!----------------------VIVADO ENVS----------------------!#
    os.environ["PATH"] = "/data2/Xilinx/Vivado/2024.2/bin:/data2/Xilinx/Vitis_HLS/2024.2/bin:" + os.environ["PATH"]

    #!----------------------CFG----------------------!#
    if backend == "vivado":
        cfg = conifer.backends.xilinxhls.auto_config(granularity="full")
        cfg["XilinxPart"] = "xcvu13p-flga2577-2-e"
        cfg['InputPrecision'] = f"ap_fixed<{precision},1,AP_RND_CONV,AP_SAT>"
        cfg['ThresholdPrecision'] = f"ap_fixed<{precision},1,AP_RND_CONV,AP_SAT>"
        cfg['ScorePrecision'] =  "ap_fixed<11,4,AP_RND_CONV,AP_SAT>"


    elif backend == "py":
        cfg = {"backend": "py", "output_dir": "dummy", "project_name": "dummy", "Precision": "float"}

    elif backend == "cpp":
        cfg = conifer.backends.cpp.auto_config()
        cfg["Precision"] = f"ap_fixed<{precision},1,AP_RND_CONV,AP_SAT>"
        cfg["score_precision"] = "ap_fixed<11,4,AP_RND_CONV,AP_SAT>"

    cfg["OutputDir"] = f"results/q_{precision}/conifer_model_{backend}"


    #!----------------------Load Model----------------------!#
    xgb_model = xgb.Booster()
    xgb_model.load_model(model)

    #!----------------------Convert Model----------------------!#
    hls_model = conifer.converters.convert_from_xgboost(xgb_model, cfg)
    print(f"Model Convered to {backend}")
    hls_model.compile()
    print("Model compiled")
    if build:
        hls_model.build()
        print("Model built")

    #!----------------------Predict----------------------!#
    if predict:
        raw_func = lambda x: np.log(x / (1 - x)) / 8
        xgb_preds = xgb_model.predict(dmatrix)
        xgb_preds = raw_func(xgb_preds)

        hls_preds = hls_model.decision_function(dmatrix.get_data().toarray())
        hls_preds = hls_preds/8
        return df_test, cfg, xgb_model, xgb_preds, hls_model, hls_preds
    else:
        return df_test, cfg, xgb_model, hls_model



# %%
def save(cfg, hls_model, precision):
    import json
    hls_model.save(f"results/q_{precision}/conifer_model.json")
    with open(f"results/q_{precision}/conifer_conf.json", "w") as f:
        f.write(json.dumps(cfg, indent=4))


for q in [8]:
    convert("vivado", q, predict=False, build=True)
    df_test, cfg, xgb_model, xgb_preds, hls_model, hls_preds = convert("cpp", q, predict=True)
    save(cfg, hls_model,q)
    bins = np.linspace(-1, 1, 100)

    fig,ax=plt.subplots()
    ax.hist(xgb_preds[df_test["TkEle_label"]==1], bins=bins, histtype="step", label="XGBoost-sig", density=True, linewidth=2)
    ax.hist(xgb_preds[df_test["TkEle_label"]==0], bins=bins, histtype="step", label="XGBoost-bkg", density=True, linewidth=2)
    ax.hist(hls_preds[df_test["TkEle_label"]==1], bins=bins, histtype="step", label="HLS-sig", density=True, linestyle="--", linewidth=2)
    ax.hist(hls_preds[df_test["TkEle_label"]==0], bins=bins, histtype="step", label="HLS-bkg", density=True, linestyle="--", linewidth=2)
    ax.set_xlabel("Score")
    ax.set_ylabel("Density")
    plt.legend()
    hep.cms.text("Phase-2 Simulation Preliminary", fontsize=18, ax = ax)
    hep.cms.lumitext("PU 200 (14 TeV)", fontsize=18, ax = ax)
    fig.savefig(f"results/q_{q}/plots/hls_vs_xgb.png")
    fig.savefig(f"results/q_{q}/plots/hls_vs_xgb.pdf")


# %%
