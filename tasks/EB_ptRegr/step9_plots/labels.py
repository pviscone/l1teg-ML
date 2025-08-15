import mplhep as hep
import hist
import numpy as np
import os
hep.style.use("CMS")

os.makedirs("plots", exist_ok=True)

feature_labels = {
    "idScore": r"$\mathrm{ID \ Score}$",
    "caloEta": r"$| \eta^{\text{Calo}} |$",
    'caloTkAbsDphi': r"$| \Delta \phi_{caloTk} |$",
    'hwTkChi2RPhi': r"$\chi^2_{tk}$ [a.u.]",
    'caloPt': r"$p_T^{Calo}$ [GeV]",
    'caloSS': r"$E_{2 \times 5}/E_{5 \times 5}$",
    'caloTkPtRatio': r"$p_T^{Calo}/p_T^{Tk}$"
}

feature_map = {
    "idScore": lambda x: x,
    "caloEta": lambda x: np.abs(x),
    'caloTkAbsDphi':  lambda x: x,
    'hwTkChi2RPhi': lambda x: x,
    'caloPt': lambda x: x,
    'caloSS': lambda x: x,
    'caloTkPtRatio': lambda x: x
}

feature_bins = {
    "idScore": hist.axis.Regular(20, -0.7, 1),
    "caloEta": hist.axis.Regular(10, 0, 1.5),
    "caloTkAbsDphi": hist.axis.Regular(30, 0, 0.3),
    "hwTkChi2RPhi": hist.axis.Regular(30, 0, 15),
    "caloPt": hist.axis.Regular(30, 0, 100),
    "caloSS": hist.axis.Regular(10, 0.4, 1),
    "caloTkPtRatio": hist.axis.Regular(80, 0, 50)
}
target = "TkEle_Gen_ptRatio"

target_map = lambda x: 1/x

target_label = r"$p_T^{Gen}/p_T^{Calo}$"

target_bins = hist.axis.Regular(30, 0, 4)




