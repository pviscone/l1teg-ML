from collections import OrderedDict
import numpy as np
import cmgrdf_cli.defaults

# The first matched pattern will be used
cmgrdf_cli.defaults.name_defaults = OrderedDict({
    "(.*)_(.*)_(?:ptRatio)": dict(
        bins=(40, 0, 2),
        label="($1) ($2) $p_{T}$ ratio",
    ),
    "(.*)_(?:pt|invPt)(.*)": dict(
        bins=(40, 0, 80),
        label="($1) ($2) $p_{T}$ [GeV]",
    ),
    "(.*)_(dVz)(.*)": dict(
        bins=(40, 0, 1.2),
        label="($1) ($2) |$\Delta$z| [cm]",
    ),
    "(.*)_(n)(.*)": dict(
        bins=(10, 0, 10),
        label="($1) # ($2)",
    ),
    "(.*)NMatch": dict(
        bins=(10, 0, 10),
        label="($1) NMatch",
    ),
    "(n)(.*)": dict(
        bins=(10, 0, 10),
        label="# ($1)",
    ),
})

cmgrdf_cli.defaults.histo1d_defaults["density"] = True