from cmgrdf_cli.flows import Tree
from CMGRDF import Define, Cut, ReDefine
from CMGRDF.collectionUtils import DefineSkimmedCollection



def flow(obj_name="TkEleL2", region="EB"):

    if obj_name == "TkEleL2":
        eta = "TkEleL2_caloEta"
        phi = "TkEleL2_caloPhi"
    elif obj_name == "TkEmL2":
        eta = "TkEmL2_eta"
        phi = "TkEmL2_phi"
    else:
        eta = f"{obj_name}_eta"
        phi = f"{obj_name}_phi"


    if region=="EB":
        region_sel = lambda x : f"abs({x}) < 1.479"
    elif region=="EE":
        region_sel = lambda x : f"abs({x}) > 1.479 && abs({x}) < 2.4"
    else:
        raise ValueError("region must be either EB or EE")

    #? ----------------- Obj Selection ---------------- #
    # Select in barrel and tight ID

    tree = Tree()


    tree.add("gen", [
        DefineSkimmedCollection("GenEl", mask=region_sel("GenEl_eta")),
        Cut(f"nGenEl_{region}>0", "nGenEl > 0", samplePattern="(?!MinBias).*"),
        Cut("nGenEl==0", "nGenEl== 0", samplePattern="MinBias.*"),
        Define(f"{obj_name}_idScore", "RVecF(nTkEleL2,0.)", eras=["v131Xv3O"]),
        Define("nEvents", "1.", plot="gen"),
    ])

    tree.add(f"{obj_name}Loose",[
        DefineSkimmedCollection(obj_name, mask = region_sel(eta)),
        Cut(f"n{obj_name}_{region}>0", f"n{obj_name} > 0")
    ], parent="gen")

    tree.add(f"{obj_name}Tight",[
        DefineSkimmedCollection(obj_name, mask = f"({obj_name}_hwQual & 2)==2 && {region_sel(eta)}"),
        Cut(f"n{obj_name}Tight>0", f"n{obj_name} > 0")
    ], parent="gen")


    tree.add_to_all("{leaf}_All",[
        #? ----------------- Obj background ---------------- #
        #Take only the highest pT object in the event
        DefineSkimmedCollection(obj_name, indices="RVecI({(int) ROOT::VecOps::ArgMax(<obj_name>_pt)})".replace("<obj_name>", obj_name), samplePattern="MinBias.*", plot=f"{obj_name}Bkg"),
        ReDefine("GenEl_pt", f"RVecF(nTkEleL2,0.)", samplePattern="MinBias.*"),

        #? ----------------- Obj signal ---------------- #
        # Match to gen (dR < 0.1) and select the highest pT object
        #Indexes to GenEl collection
        Define(f"{obj_name}_genIdx", f"get_dRmatchIdx({eta}, {phi}, GenEl_caloeta, GenEl_calophi)"),

        #Remove unmatched objects
        DefineSkimmedCollection(obj_name, mask=f"{obj_name}_genIdx!=-1"),

        #At least one matched object
        Cut(f"n{obj_name} match>0", f"n{obj_name} > 0"),

        #Select the highest pT object per different GenEl
        DefineSkimmedCollection(obj_name, mask = f"maskMaxPerGroup({obj_name}_pt, {obj_name}_genIdx)"),

        #Select only the Gen that has a matched object
        DefineSkimmedCollection("GenEl", indices=f"{obj_name}_genIdx"),
    ], samplePattern="(?!MinBias).*")

    return tree