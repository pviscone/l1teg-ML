from cmgrdf_cli.flows import Tree
from CMGRDF import Define, Cut, ReDefine
from CMGRDF.collectionUtils import DefineSkimmedCollection, DefineP4, DefineFromCollection


def flow():
    tree = Tree()
    tree.add("noRegress", [DefineSkimmedCollection("GenEl", mask="GenEl_prompt==2"),
                      Cut("nGenElBarrel==2", "Sum(abs(GenEl_caloeta)<1.479)==2", samplePattern="(?!MinBias).*")])

    tree.add("regressed", [
        ReDefine("TkEleL2_pt", "TkEleL2_ptCorr")
    ], parent=["noRegress"])


    tree.add("main_{leaf}",[

        #!------------------ TkEle ------------------!#
        DefineSkimmedCollection("TkEleL2", mask="abs(TkEleL2_caloEta)<1.479"),
        Define("TkEleMask", "TkEleL2_idScore>-0.3 && TkEleL2_pt>4", plot="noSelection"),
        DefineSkimmedCollection("TkEleL2", mask="TkEleMask"),
        Cut("2tkEle","nTkEleL2>=2", plot="2tkEle_pt3_score-0.3"),

        #!------------------ OS ------------------!#
        Define("DPCandidatesIdxs", "makeDPCandidateOSIdx(TkEleL2_charge)"),
        Cut("1OS pair", "DPCandidatesIdxs.first.size()>0"),
        DefineSkimmedCollection("DPCandidates_l1", "TkEleL2", indices="DPCandidatesIdxs.first"),
        DefineSkimmedCollection("DPCandidates_l2", "TkEleL2", indices="DPCandidatesIdxs.second"),
        Define("DPCandidates_l1_p4", "makeP4(DPCandidates_l1_pt, DPCandidates_l1_caloEta, DPCandidates_l1_caloPhi, 0.0005)"),
        Define("DPCandidates_l2_p4", "makeP4(DPCandidates_l2_pt, DPCandidates_l2_caloEta, DPCandidates_l2_caloPhi, 0.0005)"),
        Define("DPCandidates_p4", "DPCandidates_l1_p4+DPCandidates_l2_p4"),
        Define("DPCandidates_mass", "getMasses(DPCandidates_p4)"),
        Define("DPCandidates_pt", "getPts(DPCandidates_p4)"),
        Define("DPCandidates_dz", "abs(DPCandidates_l1_vz-DPCandidates_l2_vz)"),
        Define("DPCandidates_dphi", "ROOT::VecOps::DeltaPhi(DPCandidates_l1_caloPhi, DPCandidates_l2_caloPhi)"),
        Define("DPCandidates_deta", "abs(DPCandidates_l1_caloEta-DPCandidates_l2_caloEta)"),
        Define("DPCandidates_dR", "ROOT::VecOps::sqrt(DPCandidates_deta*DPCandidates_deta+DPCandidates_dphi*DPCandidates_dphi)", plot="OSPair"),
        DefineSkimmedCollection("DPCandidates", mask="DPCandidates_dz<0.65"),
        Cut(">=1 DPCandidate (dz<0.65)", "nDPCandidates>0", plot="dZCut"),

        #!------------------ Gen Matching ------------------!#
        DefineSkimmedCollection("DPCandidates", mask=f"""
                (match_mask(DPCandidates_l1_caloEta, DPCandidates_l1_caloPhi, GenEl_caloeta[0], GenEl_calophi[0]) &&
                match_mask(DPCandidates_l2_caloEta, DPCandidates_l2_caloPhi, GenEl_caloeta[1], GenEl_calophi[1])) ||
                (match_mask(DPCandidates_l1_caloEta, DPCandidates_l1_caloPhi, GenEl_caloeta[1], GenEl_calophi[1]) &&
                match_mask(DPCandidates_l2_caloEta, DPCandidates_l2_caloPhi, GenEl_caloeta[0], GenEl_calophi[0]))
        """),
        Cut(">=1 DPCandidate (matched to GenEl)", "nDPCandidates>0", plot="matchingGenCut"),


        #Define("_argmaxdphi", "ArgMax(DPCandidates_dphi)"),
        #DefineFromCollection("DPCandidate", "DPCandidates", index="_argmaxdphi", plot="B2B_selection"),
        #Cut("CompositeIDCut", "DPCandidate_score_1>0 && DPCandidate_score_2>0", plot="ScoreCut"),
        #Cut("dphiCut", "DPCandidate_dphi>2.5", plot="dphiCut"),
    ], parent=["regressed", "noRegress"])
    return tree
