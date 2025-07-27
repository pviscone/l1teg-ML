from cmgrdf_cli.flows import Tree
from CMGRDF import Define, Cut, ReDefine
from CMGRDF.collectionUtils import DefineSkimmedCollection, DefineP4, DefineFromCollection


def flow(pt="pt"):
    tree = Tree()

    redefine=[]
    if pt!="pt":
        redefine+=[
            ReDefine("TkEleL2_pt", f"TkEleL2_{pt}")
        ]
    
    tree.add("base",[
        *redefine,
        #!------------------ GenEle ------------------!#
        Cut("nGenElBarrel>=2", "Sum(abs(GenEl_caloeta)<1.479)>=2", samplePattern="(?!MinBias).*"),

        #!------------------ TkEle ------------------!#
        DefineSkimmedCollection("TkEleL2", mask="abs(TkEleL2_caloEta)<1.479"),
        Define("TkEleMask", "TkEleL2_idScore>-0.3 && TkEleL2_pt>3", plot="noSelection"),
        DefineSkimmedCollection("TkEleL2", mask="TkEleMask"),
        Cut("2tkEle","nTkEleL2>=2", plot="2tkEle_pt3_score-0.3"),

        #!------------------ OS ------------------!#
        Define("DPCandidatesIdxs", "makeDPCandidateOSIdx(TkEleL2_charge)"),
        Cut("1OS pair", "DPCandidatesIdxs.first.size()>0"),
        DefineSkimmedCollection("DPEle1", "TkEleL2", indices="DPCandidatesIdxs.first"),
        DefineSkimmedCollection("DPEle2", "TkEleL2", indices="DPCandidatesIdxs.second"),
        Define("DPEle1_p4", "makeP4(DPEle1_pt, DPEle1_caloEta, DPEle1_caloPhi, 0.0005)"),
        Define("DPEle2_p4", "makeP4(DPEle2_pt, DPEle2_caloEta, DPEle2_caloPhi, 0.0005)"),
        Define("DPCandidates_p4", "DPEle1_p4+DPEle2_p4"),
        Define("DPCandidates_mass", "getMasses(DPCandidates_p4)"),
        Define("DPCandidates_pt", "getPts(DPCandidates_p4)"),
        Define("DPCandidates_dz", "abs(DPEle1_vz-DPEle2_vz)"),
        Define("DPCandidates_dphi", "ROOT::VecOps::DeltaPhi(DPEle1_caloPhi, DPEle2_caloPhi)"),
        Define("DPCandidates_deta", "abs(DPEle1_caloEta-DPEle2_caloEta)"),
        Define("DPCandidates_dR", "ROOT::VecOps::sqrt(DPCandidates_deta*DPCandidates_deta+DPCandidates_dphi*DPCandidates_dphi)"),
        Define("DPCandidates_score_1", "DPEle1_idScore"),
        Define("DPCandidates_score_2", "DPEle2_idScore"),
        Define("DPCandidates_pt_1", "DPEle1_pt"),
        Define("DPCandidates_pt_2", "DPEle2_pt", plot="OSPair"),
        Define("DPsMask", "DPCandidates_dz<0.65"),
        DefineSkimmedCollection("DPCandidates", mask="DPsMask"),
        Cut(">=1 DPCandidate (dz<0.65)", "DPCandidates_dz.size()>0", plot="dZCut"),
        #Define("_argmaxdphi", "ArgMax(DPCandidates_dphi)"),
        #DefineFromCollection("DPCandidate", "DPCandidates", index="_argmaxdphi", plot="B2B_selection"),
        #Cut("CompositeIDCut", "DPCandidate_score_1>0 && DPCandidate_score_2>0", plot="ScoreCut"),
        #Cut("dphiCut", "DPCandidate_dphi>2.5", plot="dphiCut"),
    ])
    return tree
