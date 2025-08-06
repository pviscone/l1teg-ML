from collections import OrderedDict
import numpy as np
import cmgrdf_cli.defaults

# The first matched pattern will be used
cmgrdf_cli.defaults.name_defaults = OrderedDict({
    "(.*)_charge(.*)": dict(
        bins=(5, -2, 2),
        label="($1) ($2) charge",
    ),
    "(.*)_ptRatio(.*)": dict(
        bins=(50, 0.8, 2),
        label="($1) ($2) $p_{T}$ Ratio",
    ),
    "(.*)_pt(.*)": dict(
        bins=(35, 0, 40),
        label="($1) ($2) $p_{T}$ [GeV]",
    ),
    "(.*)_mass(.*)": dict(
        bins=(60,0.5,40.5),
        label="($1) ($2) m [GeV]",
    ),
    "(.*)_dphi(.*)": dict(
        bins=(35, 0, np.pi),
        label="($1) ($2) $\Delta \phi$",
    ),
    "(.*)_dz(.*)": dict(
        bins=(35, 0, 1),
        label="($1) ($2) |dz| [cm]",
    ),
    "(.*)_dR(.*)": dict(
        bins=(40, 0, 3),
        label="($1) ($2) $\Delta$R",
    ),
    "(.*)_deta(.*)": dict(
        bins=(35, 0, 1),
        label="($1) ($2) $\Delta \eta$",
    ),
    "(.*)_dPhi(.*)": dict(
        bins=(35, 0, np.pi),
        label="($1) ($2) $\Delta \phi$",
    ),
    "(.*)_eta(.*)": dict(
        bins=(35, -2., 2.),
        label="($1) ($2) $\eta$",
    ),
    "(.*)_phi(.*)": dict(
        bins=(35, -3.14, 3.14),
        label="($1) ($2) $\phi$",
    ),
    "(.*)_vz(.*)": dict(
        bins=(35, -10, 10),
        label="($1) ($2) vz",
    ),
    "(.*)_score(.*)": dict(
        bins=(35, -1., 1.),
        label="($1) ($2) score",
    ),
})

cmgrdf_cli.defaults.histo1d_defaults["density"] = True
#cmgrdf_cli.defaults.histo1d_defaults["log"] = "y"


