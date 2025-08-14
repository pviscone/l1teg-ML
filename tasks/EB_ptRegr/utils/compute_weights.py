#%%
import sys
import numpy as np
sys.path.append("../utils")
import hist

# %%

def get_h(values, bins, arr):
    bin_idxs = np.digitize(arr, bins[:-1])-1
    res = values[bin_idxs]
    return res

def cut_and_compute_weights(df, genpt_, pt_, genpt_bins=None, ptcut=0, genmax=100):
    df = df[df[genpt_] < genmax]
    if genpt_bins is None:
        genpt_bins = np.linspace(1, 100, 3)
    genpt_bins = genpt_bins[genpt_bins >= ptcut]
    df = df[df[genpt_] >= ptcut]

    fine_bins = np.linspace(genpt_bins[0], genpt_bins[-1], 200)
    genpt_h = hist.Hist(hist.axis.Variable(fine_bins))
    unbalance_h = hist.Hist(hist.axis.Variable(fine_bins))
    genpt_h.fill(df[genpt_].values)
    unbalance_h.fill(fine_bins)
    unbalance_h = unbalance_h/(genpt_h/genpt_h.integrate(0))

    mad_h = hist.Hist(hist.axis.Variable(genpt_bins))
    var_h = hist.Hist(hist.axis.Variable(genpt_bins))
    for genpt_min, genpt_max in zip(genpt_bins[:-1], genpt_bins[1:]):
        center = (genpt_min + genpt_max) / 2
        mask = (df[genpt_] >= genpt_min) & (df[genpt_] < genpt_max)
        df_mask = df[mask]
        mad_h[hist.loc(center)] = 1/np.median(np.abs(1-df_mask[genpt_]/df_mask[pt_]))
        var_h[hist.loc(center)] = 1/np.sum(((1-df_mask[genpt_]/df_mask[pt_])**2)/(len(df_mask) - 1))

    df["VARw"]=get_h(var_h.values(), var_h.axes[0].edges, df[genpt_].values)
    df["RESw"]=get_h(mad_h.values(), mad_h.axes[0].edges, df[genpt_].values)
    df["BALw"]=get_h(unbalance_h.values(), unbalance_h.axes[0].edges, df[genpt_].values)
    df["BALw"]= len(df)*df["BALw"]/df["BALw"].sum()
    df["wTot"]=df["RESw"]*df["BALw"]
    df["w2Tot"]=df["VARw"]*df["BALw"]
    return df


def flat_w(arr, ref_arr, weight=1, bins=None):
    if bins is None:
        bins = np.arange(1, 120, 0.25)

    hist_arr = hist.Hist(hist.axis.Variable(bins), storage=hist.storage.Weight())
    hist_ref = hist.Hist(hist.axis.Variable(bins), storage=hist.storage.Weight())
    hist_arr.fill(arr)
    hist_ref.fill(ref_arr, weight=weight)
    den = hist_arr.values()
    den[den==0]=1
    hist_arr = hist_ref/den
    res=get_h(hist_arr.values(), hist_arr.axes[0].edges, arr)
    return res*len(ref_arr)/np.sum(res)



