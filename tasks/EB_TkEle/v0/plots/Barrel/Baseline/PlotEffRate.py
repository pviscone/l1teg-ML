import copy
from flows.Barrel.Baseline.FlowBaseline import objs as objects
import plots.defaults

from cmgrdf_cli.plots import Hist
objs = copy.deepcopy(objects)
if "TkEleL2" in objs:
    objs.remove("TkEleL2")
    objs.append("TkEleLoose")
    objs.append("TkEleTight")


#!TO use with flows.Barrel.Baseline.FlowBaseline
def plots(objs=objs):
    objs_bkg = {f"{obj}Bkg":[
        Hist(f"{obj}_pt", f"{obj}_pt"),
        Hist(f"n{obj}", f"n{obj}"),
        ]
        for obj in objs
    }

    return {
        "main":[
            Hist("GenEl_pt", "GenEl_pt"),
            Hist("GenEl_eta", "GenEl_eta"),
            Hist("nEvents", "nEvents"),
        ],
        **objs_bkg
    }