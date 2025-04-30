import os
import pandas as pd

from CMGRDF import Define
from CMGRDF.collectionUtils import DefineSkimmedCollection

from cpp import load_conifer
from flows.Barrel.TkEle import FlowBase as base


def flow(bdt_path=None, n_bits=8):
    assert bdt_path is not None, (
        "bdt_path (path to the folder containing the q_<nbits>/conifer_model.json file) must be provided"
    )
    assert isinstance(n_bits, int)

    tree = base.flow()

    if not os.path.exists(os.path.join(bdt_path, f"q_{n_bits}")):
        raise ValueError(f"BDT model for {n_bits} bits not found in {bdt_path}")

    conifer_model_path = os.path.join(bdt_path, f"q_{n_bits}/conifer_model.json")

    load_conifer.declare(conifer_model_path, n_bits)

    scaler = pd.read_json(os.path.join(bdt_path, f"q_{n_bits}/scaler.json"))
    features = scaler["feature_name"].values.tolist()
    scale_steps = []
    for feature in features:
        df = scaler[scaler["feature_name"] == feature]
        scale_steps.append(
            Define(
                f"_scaled_{feature}",
                f"bitscale({feature}, {float(df['inf'].iloc[0])}, {float(df['min'].iloc[0])}, {int(df['bit_shift'].iloc[0])})",
            )
        )

    features_string = [f"_scaled_{feature}" for feature in features]
    features_string = ",".join(features_string)
    tree.add(
        "eval_bdt",
        [
            *scale_steps,
            Define(
                "TkEle_score",
                "bdt_evaluate({<feats>})".replace("<feats>", features_string),
            ),
            DefineSkimmedCollection("TkEle", mask="maskMaxPerGroup(TkEle_score, TkEle_CryClu_idx)"),
        ],
        parent="matching",
    )
    return tree
