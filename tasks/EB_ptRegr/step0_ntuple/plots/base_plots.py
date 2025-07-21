from plots import base_defaults
from cmgrdf_cli.plots import Hist, Hist2D

plots={
    "main":[

        Hist("GenEl_nMatch", "GenEl_nMatch"),
        Hist("nTkEleL2", "nTkEleL2"),

        Hist("TkEle_Gen_dVz", "TkEle_Gen_dVz"),

        Hist("TkEle_Gen_ptRatio", "TkEle_Gen_ptRatio"),
        Hist("TkEle_GenMultipleMatch_ptRatio", "TkEle_pt[TkEle_Gen_nMatch>=2]/TkEle_Gen_pt[TkEle_Gen_nMatch>=2]"),
        Hist2D("TkEle_Gen_nMatch:TkEle_Gen_dVz", "TkEle_Gen_nMatch", "TkEle_Gen_dVz"),
        Hist2D("TkEle_Gen_nMatch:TkEle_Gen_ptRatio", "TkEle_Gen_nMatch", "TkEle_Gen_ptRatio"),
        Hist2D("TkEle_Gen_dVz:TkEle_Gen_ptRatio", "TkEle_Gen_dVz", "TkEle_Gen_ptRatio"),
        Hist2D("TkEle_GenMultipleMatch_dVz:TkEle_GenMultipleMatch_ptRatio", "abs(TkEle_vz[TkEle_Gen_nMatch>=2]-TkEle_Gen_vz[TkEle_Gen_nMatch>=2])", "TkEle_pt[TkEle_Gen_nMatch>=2]/TkEle_Gen_pt[TkEle_Gen_nMatch>=2]"),
        Hist2D("TkEle_in_caloTkNMatch:TkEle_Gen_nMatch", "TkEle_in_caloTkNMatch", "TkEle_Gen_nMatch", eras=["151Xv0pre4_TkElePtRegr_dev"]),
        Hist2D("TkEle_in_caloTkNMatch:TkEle_Gen_ptRatio", "TkEle_in_caloTkNMatch", "TkEle_Gen_ptRatio", eras=["151Xv0pre4_TkElePtRegr_dev"]),
    ]
}
