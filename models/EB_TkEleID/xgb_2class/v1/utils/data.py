import os
import uproot
import numpy as np
import awkward as ak
import pandas as pd
import xgboost as xgb
from tqdm import tqdm
from sklearn.utils.class_weight import compute_sample_weight
from bithub.quantizers import mp_xilinx

#!--------------------------------------------------------------------------!#


def compute_class_weights(*dfs):
    lens = [len(df) for df in dfs]
    max_len = max(lens)
    weights = [max_len / len_ for len_ in lens]
    return (*weights,)


def select_columns(*dfs, columns):
    res = [df[columns] for df in dfs]
    return (*res,)


#!--------------------------------------------------------------------------!#


def generate_paths(P0, tags, names):
    if not isinstance(names, list | tuple):
        names = [names]
    if not isinstance(tags, list | tuple):
        tags = [tags] * len(names)
    assert len(tags) == len(names), "tags and names must have the same length"
    return [
        os.path.join(P0.format(tag=tag), f"{name}.root")
        for tag, name in zip(tags, names, strict=False)
    ]


def load_df(*files, branches=None):
    res = []
    for f in (pbar := tqdm(files, desc="Loading files and converting to DataFrame")):
        pbar.set_postfix_str(f.split("/")[-1])
        events = uproot.open(f)["Events"].arrays(branches)
        events["TkEle_ev_idx"] = ak.ones_like(events[events.fields[0]]) * np.arange(
            len(events)
        )
        events = {k: ak.ravel(events[k]).to_numpy() for k in events.fields}
        df = pd.DataFrame(events)
        df.attrs["name"] = f.split("/")[-1].split(".")[0]
        res.append(df)
    return (*res,)


def normalize_weight(*dfs, key=None, kind="entries"):
    if not isinstance(key, list | tuple):
        key = [key] * len(dfs)

    res = []
    for weight_key, df in zip(key, dfs, strict=False):
        df = df[df[weight_key] != 0]
        w = df[weight_key]
        w = w / w.sum()

        if kind == "normalize":
            df.loc[:, weight_key] = w
        elif kind == "entries":
            df.loc[:, weight_key] = w * len(df)
        res.append(df)
    return (*res,)


def df_to_DMatrix(
    *dfs,
    features=None,
    y=None,
    weight=None,
    class_weights=None,
    bitscaler=None,
    ap_fixed=None,
):
    res = []

    if features is None:
        features = dfs[0].columns

    if y in features:
        features.remove(y)

    if weight is not None and weight in features:
        features.remove(weight)

    for df in dfs:
        if isinstance(df, list | tuple):
            df = pd.concat(df)

        labels = df[y]
        weights = df[weight]
        df = df[features]

        if bitscaler is not None:
            df = bitscaler.apply(df)

        if ap_fixed is not None:
            args = []
            for el in ap_fixed:
                if not isinstance(el, str):
                    el = str(el)
                else:
                    el = f"'{el}'"
                args.append(el)
            args = ",".join(args)
            df = pd.DataFrame(
                mp_xilinx.mp_xilinx(df, f"ap_fixed<{args}>", convert="double")
            )

        if class_weights is None:
            class_w = np.ones(len(df))
        elif class_weights == "balanced":
            class_w = compute_sample_weight(class_weight="balanced", y=labels)
        else:
            raise ValueError(f"Invalid class_weight: {class_weights}")

        res.append(xgb.DMatrix(df, label=labels, weight=weights * class_w))
    return (*res,)


def concatenate(*dfs):
    res = []
    if len(dfs) == 1 and isinstance(dfs[0], dict):
        dfs = dfs[0]

    for df in dfs:
        if isinstance(dfs, dict):
            df_name = df
            df = dfs[df]

        if isinstance(df, list | tuple):
            df = pd.concat(df)
        if isinstance(dfs, dict):
            df.attrs["name"] = df_name
        res.append(df)
    return (*res,)
