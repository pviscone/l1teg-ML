import mplhep as hep
import hist
import numpy as np
import os
hep.style.use("CMS")

os.makedirs("plots", exist_ok=True)

features = [
    "hwCaloEta",
    'in_caloTkAbsDphi',
    'in_hwTkChi2RPhi',
    'in_caloPt',
    'in_caloSS',
    'in_caloTkPtRatio',
]

feature_labels = {
    "hwCaloEta": r"$| \eta |$ [a.u.]",
    'in_caloTkAbsDphi': r"$| \Delta \phi_{caloTk} |$ [a.u.]",
    'in_hwTkChi2RPhi': r"$\chi^2_{tk}$ [a.u.]",
    'in_caloPt': r"$p_T^{Calo}$ [GeV]",
    'in_caloSS': r"$E_{2 \times 5}/E_{5 \times 5}$",
    'in_caloTkPtRatio': r"$p_T^{Calo}/p_T^{Tk}$"
}

feature_map = {
    "hwCaloEta": lambda x: np.abs(x),
    'in_caloTkAbsDphi':  lambda x: x,
    'in_hwTkChi2RPhi': lambda x: x,
    'in_caloPt': lambda x: x,
    'in_caloSS': lambda x: x,
    'in_caloTkPtRatio': lambda x: x
}

feature_bins = {
    "hwCaloEta": hist.axis.Regular(20, 0, 340),
    "in_caloTkAbsDphi": hist.axis.Regular(30, 0, 70),
    "in_hwTkChi2RPhi": hist.axis.Regular(30, 0, 15),
    "in_caloPt": hist.axis.Regular(20, 0, 100),
    "in_caloSS": hist.axis.Regular(30, 0.3, 1),
    "in_caloTkPtRatio": hist.axis.Regular(30, 0, 50)
}

target = "TkEle_Gen_ptRatio"

target_map = lambda x: 1/x

target_label = r"$p_T^{Gen}/p_T^{Calo}$"

target_bins = hist.axis.Regular(30, 0, 4)

testA2 = "../step0_ntuple/tempA2/zsnap/era151X_ptRegr_v0_A2/base_2_ptRatioMultipleMatch05/DoubleElectron_PU200.root"
train = "../step0_ntuple/tempTrain/zsnap/era151Xv0pre4_TkElePtRegr_dev_withScaled/base_2_ptRatioMultipleMatch05/DoubleElectron_PU200.root"