from cmgrdf_cli.flows import Tree
from CMGRDF import Define, Cut, ReDefine
from CMGRDF.collectionUtils import DefineSkimmedCollection


class L1_obj:
    def __init__(self, name: str, etaVertex: str, phiVertex: str, etaCalo: str, phiCalo: str, matchAt: str): #Vertex, Calo
        self.name = name
        self.etaVertex = f"{name}_{etaVertex}" if etaVertex is not None else None
        self.phiVertex = f"{name}_{phiVertex}" if phiVertex is not None else None
        self.etaCalo = f"{name}_{etaCalo}" if etaCalo is not None else None
        self.phiCalo = f"{name}_{phiCalo}" if phiCalo is not None else None
        self.matchAt = matchAt

    @property
    def Calo(self):
        return f"{self.etaCalo}, {self.phiCalo}"

    @property
    def Vertex(self):
        return f"{self.etaVertex}, {self.phiVertex}"

    def Get(self, what):
        return getattr(self, what)

    @property
    def GetMatchVar(self):
        if self.matchAt is not None:
            return getattr(self, self.matchAt)
        else:
            raise ValueError(f"{self.name} has not matchAt attribute")


TkEleL2 = L1_obj("TkEleL2", "eta", "phi", "caloEta", "caloPhi", "Vertex")
TkEmL2 = L1_obj("TkEmL2", None, None, "eta", "phi", "Calo")
GenEl = L1_obj("GenEl", "eta", "phi", "caloeta", "calophi", None)

def flow(obj_name="TkEleL2", region="EB"):
    obj = eval(obj_name)

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
        DefineSkimmedCollection("GenEl", mask=region_sel(GenEl.etaCalo)),
        Cut(f"nGenEl_{region}>0", "nGenEl > 0", samplePattern="(?!MinBias).*"),
        Cut("nGenEl==0", "nGenEl== 0", samplePattern="MinBias.*"),
        Define(f"{obj.name}_idScore", "RVecF(nTkEleL2,0.)", eras=["v131Xv3O"]),
        Define("nEvents", "1.", plot="gen"),
    ])

    tree.add(f"{obj.name}Loose",[
        DefineSkimmedCollection(obj.name, mask = region_sel(obj.etaCalo)),
        Cut(f"n{obj.name}_{region}>0", f"n{obj.name} > 0")
    ], parent="gen")

    tree.add(f"{obj.name}Tight",[
        DefineSkimmedCollection(obj.name, mask = f"({obj.name}_hwQual & 2)==2 && {region_sel(obj.etaCalo)}"),
        Cut(f"n{obj.name}Tight>0", f"n{obj.name} > 0")
    ], parent="gen")

    tree.add_to_all("{leaf}_All",[
        #? ----------------- Obj background ---------------- #
        #Take only the highest pT object in the event (NO!!!)
        #DefineSkimmedCollection(obj.name, indices="RVecI({(int) ROOT::VecOps::ArgMax(<obj.name>_pt)})".replace("<obj.name>", obj.name), samplePattern="MinBias.*", plot=f"{obj.name}Bkg"),
        ReDefine("GenEl_pt", f"RVecF(nTkEleL2,0.)", samplePattern="MinBias.*"),

        #? ----------------- Obj signal ---------------- #
        # Match to gen (dR < 0.1) and select the highest pT object
        #Indexes to GenEl collection
        Define(f"{obj.name}_genIdx", f"get_dRmatchIdx({obj.GetMatchVar}, {GenEl.Get(obj.matchAt)})"),

        #Remove unmatched objects
        DefineSkimmedCollection(obj.name, mask=f"{obj.name}_genIdx!=-1"),

        #At least one matched object
        Cut(f"n{obj.name} match>0", f"n{obj.name} > 0"),

        #Select the highest pT object per different GenEl
        DefineSkimmedCollection(obj.name, mask = f"maskMaxPerGroup({obj.name}_pt, {obj.name}_genIdx)"),

        #Select only the Gen that has a matched object
        DefineSkimmedCollection("GenEl", indices=f"{obj.name}_genIdx"),
    ], samplePattern="(?!MinBias).*")

    return tree
