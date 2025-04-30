from CMGRDF import Define
from cpp import load_clupt_h

from flows.Barrel.TkEle import FlowBase as base

def flow(pt_hist=None, bkg_pt_cut=64.0):
    assert pt_hist is not None, "pt_hist ('filepath,sig_branch,bkg_branch') must be provided"

    file, sig, bkg = pt_hist.split(",")

    tree = base.flow()

    load_clupt_h.declare(file, sig, bkg)
    tree.add(
        "reweight",
        [
            # only MinBias_train
            Define(
                "TkEle_weight",
                f"(TkEle_CryClu_pt<{bkg_pt_cut})*reweight_pt_h(TkEle_CryClu_pt)",
                samplePattern="MinBias_train",
            ),
            # all train apart from MinBias_train
            Define(
                "TkEle_weight",
                f"(TkEle_CryClu_pt<{bkg_pt_cut})*RVecF(nTkEle, 1.)",
                samplePattern="^(?!MinBias_).*_train",
            ),
            # all other samples (non train)
            Define(
                "TkEle_weight",
                "RVecF(nTkEle, 1.)",
                samplePattern=".*(?<!_train)",
            ),
        ],
        parent="matching",
    )
    return tree