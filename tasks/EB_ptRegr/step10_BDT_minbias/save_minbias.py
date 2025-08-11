import ROOT
ROOT.EnableImplicitMT()
minbias = "root://eoscms.cern.ch//eos/cms/store/cmst3/group/l1tr/pviscone/l1teg/fp_ntuples/NuGunAllEta_PU200/FP/151X_ptRegr_v0_A2/*.root"
minbias = ROOT.RDataFrame("Events", minbias)
minbias = (minbias.Define("mask", "(TkEleL2_hwQual & 2) == 2 && abs(TkEleL2_caloEta) < 1.479")
           .Filter("Sum(mask) > 0")
)
cols = minbias.GetColumnNames()
tkele_cols = [col for col in cols if col.startswith("TkEleL2_")]
for col in tkele_cols:
    minbias = minbias.Define(col.replace("TkEleL2", "TkEle"), f"{col}[mask][0]")

tkele_cols = [col.replace("TkEleL2", "TkEle") for col in tkele_cols]
minbias.Snapshot("Events", "minbias.root", tkele_cols)