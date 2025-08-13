import os
import numpy as np

base_path = os.path.dirname(os.path.abspath(__file__)).replace("/eos/home-p", "/eos/user/p")


signal_train = os.path.join(base_path, "step0_ntuple/tempSigTrain/zsnap/era151Xv0pre4_TkElePtRegr_dev/base_2_ptRatioMultipleMatch05/DoubleElectron_PU200.root")
signal_test = os.path.join(base_path, "step0_ntuple/tempSigTest/zsnap/era151X_ptRegr_v0_A2/base_2_ptRatioMultipleMatch05/DoubleElectron_PU200.root")
bkg_train = os.path.join(base_path, "step0_ntuple/tempMinBias/zsnap/era151X_ptRegr_v0_A2/base/MinBias.root")
bkg_test = "root://eoscms.cern.ch//eos/cms/store/cmst3/group/l1tr/pviscone/l1teg/fp_ntuples/NuGunAllEta_PU200_test/FP/151X_ptRegr_v0_A2/*.root"

for f in [signal_train, signal_test, bkg_train]:
    if not os.path.exists(f):
        raise FileNotFoundError(f"File {f} does not exist. run step 0_ntuples first.")

eta_ = "TkEle_caloEta"
genpt_ = "TkEle_Gen_pt"
pt_ = "TkEle_in_caloPt"
ptratio_dict = {"NoRegression": "TkEle_Gen_ptRatio",
                "Regressed": "TkEle_regressedPtRatio",}

metric = "L1"
quant = 10
out_cut = 2
q_out = (11,2)

def get_loss_and_eval_metric(metric):
    if metric == "L1":
        loss = "reg:absoluteerror"
        eval_metric = "mae"
        w = "wTot"
    elif metric == "L2":
        loss = "reg:squarederror"
        eval_metric = "rmse"
        w = "w2Tot"
    else:
        raise ValueError("Unknown metric. Use 'L1' or 'L2'.")
    return loss, eval_metric, w

loss, eval_metric, w = get_loss_and_eval_metric(metric)

def scale(df):
    tkRPhi_bins = np.array([0.0, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 10.0, 15.0, 20.0, 35.0, 60.0, 200.0])
    df["caloEta"] = np.clip(-1+df["TkEle_caloEta"].abs(), -1, 1)
    df["caloTkAbsDphi"] = np.clip(-1 + df["TkEle_in_caloTkAbsDphi"]/2**5, -1, 1)
    df["hwTkChi2RPhi"] = np.clip(-1 + (np.digitize(df["TkEle_in_tkChi2RPhi"], tkRPhi_bins)-1)/2**3, -1, 1)
    df["caloPt"] = np.clip(-1+(df["TkEle_in_caloPt"])/2**5, -1, 1)
    df["caloRelIso"] = np.clip(-1 + df["TkEle_in_caloRelIso"]*2**3, -1, 1)
    df["caloSS"] = np.clip(-1 + df["TkEle_in_caloSS"]*2, -1, 1)
    df["tkPtFrac"] = np.clip(-1 + df["TkEle_in_tkPtFrac"]/2**3, -1, 1)
    df["caloTkNMatch"] = np.clip(-1 + df["TkEle_in_caloTkNMatch"]/2**2, -1, 1)
    df["caloTkPtRatio"] = np.clip(-1 + df["TkEle_in_caloTkPtRatio"]/2**3, -1, 1)
    df["idScore"] = np.clip(df["TkEle_idScore"],-1,1)
    return df

features_q = [
    #'TkEle_in_caloStaWP',
    #'TkEle_in_caloTkAbsDeta',
    #'TkEle_in_caloLooseTkWP',
    #'tkPtFrac',
    #'caloTkNMatch',
    #'caloRelIso',
    'idScore',
    "caloEta",
    'caloTkAbsDphi',
    'hwTkChi2RPhi',
    'caloPt',
    'caloSS',
    'caloTkPtRatio',
]

xgbmodel="../models/xgb_model_L1_q10_out11_2.json"
conifermodel = xgbmodel.replace("xgb", "conifer")
init_pred = 512 * 2**-9  # Initial prediction value for the BDT model