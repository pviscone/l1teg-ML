from collections import OrderedDict
import numpy as np
import cmgrdf_cli.defaults

# The first matched pattern will be used
cmgrdf_cli.defaults.name_defaults = OrderedDict({
    "(.*)_(?:pt|invPt)(.*)": dict(
        bins=(40, 0, 80),
        label="($1) ($2) $p_{T}$ [GeV]",
    ),
})

cmgrdf_cli.defaults.histo1d_defaults["density"] = True