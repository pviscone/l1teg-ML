tag = "140Xv0B9"
P0 = f"/eos/user/p/pviscone/www/L1T/l1teg/EB_pretrain/v0/zsnap/era{tag}/reweight"

features = [
    "CryClu_pt",
    "CryClu_showerShape",
    "CryClu_relIso",
    "CryClu_standaloneWP",
    "CryClu_looseL1TkMatchWP",
    "Tk_chi2RPhi",
    "Tk_ptFrac",
    "PtRatio",
    "nTkMatch",
    "absdeta",
    "absdphi",
]

auxiliary = ["GenEle_pt", "GenEle_idx", "label", "weight", "sumTkPt", "CryClu_idx", "CryClu_eta"]

samples = [
    "DoubleElectron_PU200_train",
    "DoubleElectron_PU200_test",
    "MinBias_train",
    "MinBias_test",
]


features = [f"TkEle_{f}" for f in features]
auxiliary = [f"TkEle_{f}" for f in auxiliary]