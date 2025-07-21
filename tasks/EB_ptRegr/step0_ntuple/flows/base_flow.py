from cmgrdf_cli.flows import Tree
from CMGRDF import Define, Cut
from CMGRDF.collectionUtils import DefineSkimmedCollection


def flow(dR_genMatch = 0.1):
    tree = Tree()
    tree.add("base",[
        DefineSkimmedCollection("GenEl", mask = "abs(GenEl_caloeta)<1.479"),
        Cut("EB", "nGenEl==2"),
        DefineSkimmedCollection("TkEleL2", mask="abs(TkEleL2_caloEta)<1.479", eras = ["140Xv0C1"]),             #on PU0 sample
        DefineSkimmedCollection("TkEleL2", mask="TkEleL2_in_caloPt>0", eras = ["151Xv0pre4_TkElePtRegr_dev"]),  #on PU200 sample (custom)
        Define("TkEleL2_genIdx", f"match(TkEleL2_caloEta, TkEleL2_caloPhi, GenEl_caloeta, GenEl_calophi, {dR_genMatch})"), #Cut also on dz?
        Define("GenEl_nMatch", "count_matched(TkEleL2_genIdx, nGenEl)"),
        DefineSkimmedCollection("TkEle", "TkEleL2", mask = "TkEleL2_genIdx!=-1"),

        DefineSkimmedCollection("TkEle_Gen", "GenEl", indices="TkEle_genIdx"),

        Define("TkEle_Gen_dVz", "abs(TkEle_vz-TkEle_Gen_vz)"),
        Define("TkEle_Gen_ptRatio", "TkEle_pt/TkEle_Gen_pt"),
        Cut("matched_tkele", "nTkEle>0", plot="matching"),
        DefineSkimmedCollection("TkEle", mask = "TkEle_Gen_dVz<0.7"),
        Cut("dvz07", "nTkEle>0", plot="dvz07"),
        DefineSkimmedCollection("TkEle", mask = "TkEle_Gen_nMatch==1 || (TkEle_Gen_nMatch>=2 && TkEle_Gen_ptRatio>0.5)"),
        Cut("ptRatioMultipleMatch05", "nTkEle>0", plot="ptRatioMultipleMatch05"),
    ])
    return tree
