#%%
import sys
import numpy as np
sys.path.append("../utils")
from file_utils import openAsDataframe
import hist

# %%

def cut_and_compute_weights(df, genpt_, pt_, genpt_bins=None, ptcut=4):
    if genpt_bins is None:
        genpt_bins = np.linspace(4, 100, 33)
    genpt_bins = genpt_bins[genpt_bins >= ptcut]
    df = df[df[genpt_] >= ptcut]
    df = df[df[genpt_] < 100]


    fine_bins = np.linspace(genpt_bins[0], genpt_bins[-1], 200)
    genpt_h = hist.Hist(hist.axis.Variable(fine_bins))
    unbalance_h = hist.Hist(hist.axis.Variable(fine_bins))
    genpt_h.fill(df[genpt_].values)
    unbalance_h.fill(fine_bins)
    unbalance_h = unbalance_h/(genpt_h/genpt_h.integrate(0))

    mad_h = hist.Hist(hist.axis.Variable(genpt_bins))
    for genpt_min, genpt_max in zip(genpt_bins[:-1], genpt_bins[1:]):
        center = (genpt_min + genpt_max) / 2
        mask = (df[genpt_] >= genpt_min) & (df[genpt_] < genpt_max)
        df_mask = df[mask]
        mad_h[hist.loc(center)] = 1/np.median(np.abs(df_mask[pt_]-df_mask[genpt_]))

    df["RESw"]=np.array([mad_h[hist.loc(v)] for v in df[genpt_].values])
    df["BALw"]=np.array([unbalance_h[hist.loc(v)] for v in df[genpt_].values])
    df["BALw"]= len(df)*df["BALw"]/df["BALw"].sum()
    df["wTot"]=df["RESw"]*df["BALw"]
    return df

#%%
if __name__ == "__main__":
    collection = "TkEle"
    genpt_ = f"{collection}_Gen_pt"
    pt_ = "TkEle_in_caloPt"

    df = openAsDataframe("DoubleElectron_PU200.root", "TkEle")
#%%
if __name__ == "__main__":
    new_df = cut_and_compute_weights(df, genpt_, pt_)
# %%
