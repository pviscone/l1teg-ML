import os
import sys
import copy

sys.path.append(os.environ["PWD"])

import uproot
import awkward as ak
import pandas as pd
import numpy as np
import hist

def _plot_rate_normal(teff, obj, path_rate, branch_rate):
    rate_hist = uproot.open(path_rate)[branch_rate].to_hist()
    teff.add(rate_hist, label=obj)
    return teff

def _plot_rate_varbins(tRate, obj, path, score, rateVar, binVar, bins, thrs, off_scaling):
    if off_scaling:
        scale_fun = eval(f"lambda online_pt: {off_scaling}")
    else:
        scale_fun = lambda x: x


    events = uproot.open(path)["Events"].arrays()
    events["obj_ev_idx"] = ak.ones_like(events[score]) * np.arange(
        len(events)
    )
    events["__obj_weight"] = ak.ones_like(events[score]) * events["weight"].to_numpy()

    events = {k: ak.ravel(events[k]).to_numpy() for k in [rateVar, score, binVar, "__obj_weight", "obj_ev_idx"]}
    df = pd.DataFrame(events)
    df_list = []
    for idx, low_edge in enumerate(bins):
        if low_edge == bins[-1]:
            new_df = df[df[binVar] >= low_edge]
        else:
            new_df = df[
                np.bitwise_and(df[binVar] >= low_edge, df[binVar] < bins[idx + 1])
            ]

        new_df = new_df[new_df[score] > thrs[idx]]
        df_list.append(new_df)
    new_df = pd.concat(df_list)
    new_df = new_df.loc[
        new_df.groupby(["obj_ev_idx"])[rateVar].idxmax()
    ].reset_index()

    h = hist.Hist(hist.axis.Regular(120, 0, 120), storage=hist.storage.Weight())
    h.fill(scale_fun(new_df[rateVar].to_numpy()), weight=new_df["__obj_weight"].to_numpy())
    tRate.add(h, label=obj)
    return tRate
