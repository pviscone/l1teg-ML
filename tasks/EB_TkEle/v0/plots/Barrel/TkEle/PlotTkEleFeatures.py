from plots import defaults
from cmgrdf_cli.plots import Hist, Hist2D

tkele_features = [
    "GenEle_pt",
    "GenEle_eta",
    "GenEle_phi",
    "GenEle_caloeta",
    "GenEle_calophi",
    "CryClu_pt",
    "CryClu_eta",
    "CryClu_phi",
    "CryClu_caloIso",
    "CryClu_showerShape",
    "Tk_pt",
    "Tk_eta",
    "Tk_phi",
    "Tk_caloEta",
    "Tk_caloPhi",
    "Tk_chi2Bend",
    "Tk_chi2RZ",
    "Tk_chi2RPhi",
    "absdeta",
    "absdphi",
    "CryClu_relIso",
    "PtRatio",
    "CryClu_standaloneWP",
    "CryClu_looseL1TkMatchWP",
    "label",
    "nTkMatch",
    "Tk_ptFrac",
    "sumTkPt",
]

def plots(th2=False, reweight=True, pt_only=False, score_only=False):
    """

    Generate a list of histograms based on the provided options.
    Parameters:
    th2 (bool): If True, generate 2D histograms in addition to 1D histograms. Default is False.
    reweight (bool): If True, generate reweighted histograms. Default is True.
    pt_only (bool): If True, save only the cluster pt. Default is False.
    """
    if pt_only:
        return {"full":[Hist("TkEle_CryClu_pt", "TkEle_CryClu_pt")]}

    if score_only:
        return {"gen":[Hist("GenEl_pt", "GenEle_pt"),
                       Hist("GenEl_eta", "GenEle_eta")],
            "full":[Hist("TkEle_score", "TkEle_score")]}

    th1_list = [Hist(f"TkEle_{feat}",f"TkEle_{feat}") for feat in tkele_features]
    th1_list_reweighted = []
    if reweight:
        th1_list_reweighted = [Hist(f"reweight_TkEle_{feat}",f"TkEle_{feat}", weight="weight*TkEle_weight") for feat in tkele_features]
        th1_list.append(Hist("TkEle_weight", "TkEle_weight"))

    n_features = len(tkele_features)
    th2_list=[]
    th2_list_reweighted=[]
    if th2:
        for i1 in range(n_features):
            for i2 in range(i1+1,n_features):
                th2_list.append(Hist2D(f"TkEle_{tkele_features[i1]}:TkEle_{tkele_features[i2]}",f"TkEle_{tkele_features[i1]}",f"TkEle_{tkele_features[i2]}"))
                if reweight:
                    th2_list_reweighted.append(Hist2D(f"reweight_TkEle_{tkele_features[i1]}:TkEle_{tkele_features[i2]}",f"TkEle_{tkele_features[i1]}",f"TkEle_{tkele_features[i2]}", weight="weight*TkEle_weight"))
    return {
        "gen" : [Hist("GenEle_pt", "GenEle_pt")],
        "full": [
            *th1_list,
            *th1_list_reweighted,
            *th2_list,
            *th2_list_reweighted,
        ]
    }


