from plots import base_defaults
from cmgrdf_cli.plots import Hist, Hist2D
import numpy as np

plots={
    "main":[
        Hist("TkEle_pt", "TkEle_pt", log="y", bins=list(np.arange(0,100,0.125)), xlim=(0,30)),
    ]
}
