#%%
import uproot
import awkward as ak
import pandas as pd

def openAsDataframe(path, collection):
    file = uproot.open(path)["Events"]
    keys = file.keys()
    keys.remove(f"n{collection}")
    arrays = file.arrays(keys)
    return pd.DataFrame({key: ak.flatten(arrays[key]) for key in keys})
# %%
