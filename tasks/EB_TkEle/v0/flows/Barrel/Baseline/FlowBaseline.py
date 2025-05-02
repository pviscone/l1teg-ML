from cmgrdf_cli.flows import Tree
from CMGRDF import Define, Cut, Dummy
from CMGRDF.collectionUtils import DefineSkimmedCollection



def create_regions(tree, name_obj, new_name=None, tight=True, region="EB"):
    if region=="EB":
        region_sel = " < 1.479"
    elif region=="EE":
        region_sel = " > 1.479"
    else:
        raise ValueError("region must be either EB or EE")


    if name_obj == "TkEleL2":
        eta = "TkEleL2_caloEta"
        phi = "TkEleL2_caloPhi"
    elif name_obj == "TkEmL2":
        eta = "TkEmL2_eta"
        phi = "TkEmL2_phi"

    if new_name is None:
        new_name = name_obj
    #? ----------------- Obj Selection ---------------- #
    # Select in barrel and tight ID
    if tight is True:
        mask = f"({name_obj}_hwQual & 2)==2 && abs({eta}) {region_sel}"
    else:
        mask = f"abs({eta}) {region_sel}"

    eta = eta.replace(name_obj, new_name)
    phi = phi.replace(name_obj, new_name)

    tree.add(f"{new_name}Sel",[
        DefineSkimmedCollection(new_name, name_obj, mask=mask),
        Cut(f"n{new_name}_EB>0", f"n{new_name} > 0")
    ], parent="gen")


    #? ----------------- Obj signal ---------------- #
    # Match to gen (dR < 0.1) and select the highest pT object
    tree.add(f"{new_name}Sig",[
        #Indexes to GenEl collection
        Define(f"{new_name}_genIdx", f"get_dRmatchIdx({eta}, {phi}, GenEl_caloeta, GenEl_calophi)"),

        #Remove unmatched objects
        DefineSkimmedCollection(new_name, mask=f"{new_name}_genIdx!=-1"),

        #At least one matched object
        Cut(f"n{new_name}>0", f"n{new_name} > 0"),

        #Select the highest pT object per different GenEl
        DefineSkimmedCollection(new_name, mask = f"maskMaxPerGroup({new_name}_pt, {new_name}_genIdx)"),

        #Select only the Gen that has a matched object
        DefineSkimmedCollection("GenEl", indices=f"{new_name}_genIdx", plot=f"{new_name}Sig"),

    ], parent=f"{new_name}Sel", samples="(?!MinBias).*")

    #? ----------------- Obj background ---------------- #
    #Take only the highest pT object in the event
    tree.add(f"{new_name}Bkg",[
        DefineSkimmedCollection(new_name, indices="RVecI({(int) ROOT::VecOps::ArgMax(<new_name>_pt)})".replace("<new_name>", new_name), plot=f"{new_name}Bkg"),
    ], parent=f"{new_name}Sel", samples="MinBias.*")

    return tree

objs = ["TkEleL2", "TkEmL2"]

def flow(objs=objs, region="EB"):
    tree = Tree()
    #! ----------------- GEN section ---------------- #

    #Select GenEl in the region of interest
    if region=="EB":
        region_sel = " < 1.479"
    elif region=="EE":
        region_sel = " > 1.479"
    else:
        raise ValueError("region must be either EB or EE")

    tree.add("gen", [
        DefineSkimmedCollection("GenEl", mask=f"abs(GenEl_eta) {region_sel}"),
        Cut(f"nGenEl_{region}>0", "nGenEl > 0", samplePattern="(?!MinBias).*"),
        Cut("nGenEl==0", "nGenEl== 0", samplePattern="MinBias.*"),
        Define("nEvents", "1."),
        Dummy(plot="gen")
    ])
    for obj in objs:
        if obj=="TkEleL2":
            tree=create_regions(tree, obj, new_name="TkEleTight", tight=True, region=region)
            tree=create_regions(tree, obj, new_name="TkEleLoose", tight=False, region=region)
        else:
            tree=create_regions(tree, obj, region=region)

    return tree