import os
import sys
import copy

sys.path.append(os.environ["ANALYSIS_DIR"])

import uproot
import awkward as ak
import pandas as pd
import numpy as np


def _plot_efficiency_normal(teff, obj, path_num, branch_num, path_den, branch_den):
    num = uproot.open(path_num)[branch_num].to_hist()
    den = uproot.open(path_den)[branch_den].to_hist()
    teff.add(num, den, label=obj)
    return teff


def _plot_efficiency_varbins(
    teff,
    obj,
    path_num,
    path_den,
    branch_den,
    score,
    variable,
    genidx,
    binVar,
    bins,
    thrs,
):
    events = uproot.open(path_num)["Events"].arrays()
    events["TkEle_ev_idx"] = ak.ones_like(events[events.fields[0]]) * np.arange(
        len(events)
    )
    events = {k: ak.ravel(events[k]).to_numpy() for k in events.fields if k!="weight"}
    df = pd.DataFrame(events)

    den = uproot.open(path_den)[branch_den].to_hist()

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
        new_df.groupby(["TkEle_ev_idx", genidx])[score].idxmax()
    ].reset_index()

    num_h = copy.deepcopy(den)
    num_h.reset()

    num_h.fill(new_df[variable].to_numpy())
    teff.add(num_h, den, label=obj)

    return teff
