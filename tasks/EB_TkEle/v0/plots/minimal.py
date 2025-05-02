import plots.defaults
from cmgrdf_cli.plots import Hist, Hist2D


def plots (obj_name="TkEleL2"):
    return{
    "main": [
        Hist("GenEl_pt", "GenEl_pt"),
        Hist("GenEl_eta", "GenEl_eta"),
        Hist(f"{obj_name}_pt", f"{obj_name}_pt"),
        Hist("nEvents", "nEvents"),
        Hist2D("TkEleL2_pt:TkEleL2_idScore", "TkEleL2_pt", "TkEleL2_idScore"),
    ],
    "full" : [
        Hist2D("GenEl_pt:TkEleL2_idScore", "GenEl_pt", "TkEleL2_idScore"),
        Hist2D("TkEleL2_pt:GenEl_pt", "TkEleL2_pt", "GenEl_pt"),
    ]
}
