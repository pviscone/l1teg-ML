from cmgrdf_cli.flows import Tree
from CMGRDF import Define, Cut
from CMGRDF.collectionUtils import DefineSkimmedCollection, DefineP4, DefineFromCollection


def flow(dR_genMatch = 0.1):
    tree = Tree()
    tree.add("base",[Cut("dummy","1")])
    return tree
