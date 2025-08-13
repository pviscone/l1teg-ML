from cmgrdf_cli.flows import Tree
from CMGRDF import Define, Cut
from CMGRDF.collectionUtils import DefineSkimmedCollection


def flow():
    tree = Tree()
    tree.add("base",[
        Cut("ZeroGenEl", "nGenEl==0"),
        DefineSkimmedCollection("TkEle", "TkEleL2", mask="abs(TkEleL2_caloEta)<1.479 && TkEleL2_pt>1"),
        Cut("nTkEle", "nTkEle>0"),
        Define("TkEle_Gen_ptRatio", "RVecF(nTkEle, 1.)"),
    ])
    return tree
