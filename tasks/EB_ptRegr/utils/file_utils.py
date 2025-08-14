#%%
import uproot
import awkward as ak
import pandas as pd
from compute_weights import cut_and_compute_weights, flat_w
import numpy as np
import sys
sys.path.append("..")
sys.path.append("../../../utils/BitHub")
from common import pt_, scale, genpt_, w
from bithub.quantizers import xilinx, mp_xilinx

def openAsDataframe(path, collection):
    file = uproot.open(path)["Events"]
    keys = file.keys()
    if f"n{collection}" in keys:
        keys.remove(f"n{collection}")
    arrays = file.arrays(keys)
    return pd.DataFrame({key: ak.flatten(arrays[key]) for key in keys})
# %%

def open_signal(filepath):
    df_sig = openAsDataframe(filepath, "TkEle")
    df_sig = df_sig[df_sig[pt_]>0]
    df_sig = scale(df_sig)
    df_sig["label"] = 1.
    df_sig["target"] = 1/df_sig["TkEle_Gen_ptRatio"].values
    df_sig = cut_and_compute_weights(df_sig, genpt_, pt_, ptcut = 0)
    return df_sig



def open_bkg(filepath, df_sig, flat_pt=True):
    df_bkg = openAsDataframe(filepath, "TkEle")
    df_bkg = df_bkg[df_bkg[pt_]>0]
    df_bkg = scale(df_bkg)
    df_bkg["TkEle_Gen_pt"]= df_bkg[pt_].values
    df_bkg["TkEle_Gen_ptRatio"] = 1.
    df_bkg["label"] = 0.
    df_bkg["target"] = 1.
    if flat_pt:
        df_bkg["BALw"] = flat_w(df_bkg[pt_].values, df_sig[pt_].values, weight = df_sig["BALw"].values)
    else:
        df_bkg["BALw"] = np.ones(len(df_bkg)) * len(df_sig) / len(df_bkg)

    df_bkg[w]=df_bkg["BALw"]
    return df_bkg



def merge_signal_bkg(df_sig, df_bkg):
    ks = list(set(df_sig.keys()).intersection(set(df_bkg.keys())))
    df = pd.concat([df_sig[ks], df_bkg[ks]])
    return df


def quantize_features(df, features_q, quant):
    df_quant = pd.DataFrame(
        mp_xilinx.mp_xilinx({k:df[k].values for k in features_q}, f'ap_fixed<{quant}, 1, "AP_RND_CONV", "AP_SAT">', convert="double")
    )
    for k in features_q:
        df[k] = df_quant[k].values
    return df

def quantize_target(df, q_out):
    t = xilinx.convert(xilinx.ap_fixed(q_out[0], q_out[1], "AP_RND_CONV", "AP_SAT")(1/df["TkEle_Gen_ptRatio"].values), "double")
    df["target"] = t
    return df
