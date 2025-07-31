from cmgrdf_cli.flows import Tree
from CMGRDF import Cut
from CMGRDF.collectionUtils import DefineSkimmedCollection


def flow():
    tree = Tree()

    tree.add("main",[
        #!------------------ TkEle ------------------!#
        DefineSkimmedCollection("TkEleL2", mask="abs(TkEleL2_caloEta)<1.479"),
        Cut("tkEle","nTkEleL2>=0"),
        DefineSkimmedCollection("TkEleL2_tight", mask="(TkEleL2_hwQual & 2)==2"),
    ])
    return tree
