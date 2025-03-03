import os
import uproot
import numpy as np
import awkward as ak
import pandas as pd
from tqdm import tqdm


def generate_paths(P0, tags, names):
    if not isinstance(names, list|tuple):
        names = [names]
    if not isinstance(tags, list|tuple):
        tags = [tags]*len(names)
    assert len(tags) == len(names), "tags and names must have the same length"
    return [os.path.join(P0.format(tag=tag), f"{name}.root") for tag, name in zip(tags, names, strict=False)]

def load_df(*files, branches=None):
    res = []
    for f in (pbar:=tqdm(files, desc="Loading files and converting to DataFrame")):
        pbar.set_postfix_str(f.split("/")[-1])
        events = uproot.open(f)["Events"].arrays(branches)
        events["TkEle_ev_idx"]=ak.ones_like(events[events.fields[0]])*np.arange(len(events))
        events = {k:ak.ravel(events[k]).to_numpy() for k in events.fields}
        df = pd.DataFrame(events)
        df.attrs['name'] = f.split("/")[-1].split(".")[0]
        res.append(df)
    return (*res,)


def normalize_weight(*dfs, key=None, kind="entries"):
    if not isinstance(key, list|tuple):
        key = [key]*len(dfs)

    res = []
    for weight_key, df in zip(key, dfs, strict=False):
        df=df[df[weight_key]!=0]
        w = df[weight_key]
        w = w/w.sum()

        if kind == "normalize":
            df.loc[:, weight_key] = w
        elif kind == "entries":
            df.loc[:, weight_key] = w*len(df)
        res.append(df)
    return (*res,)

def compute_class_weights(*dfs):
    lens = [len(df) for df in dfs]
    max_len = max(lens)
    weights = [max_len/len_ for len_ in lens]
    return (*weights,)

def select_columns(*dfs, columns):
    res = [df[columns] for df in dfs]
    return (*res,)