from plots import base_defaults
from cmgrdf_cli.plots import Hist, Hist2D

plots={
    "main":[
        Hist("TkEleL2_charge", "TkEleL2_charge"),
        Hist("TkEleL2_vz", "TkEleL2_vz"),
        Hist("TkEleL2_score", "TkEleL2_idScore"),
        Hist("TkEleL2_pt", "TkEleL2_pt"),
        Hist("TkEleL2_eta", "TkEleL2_caloEta"),
        Hist("TkEleL2_phi", "TkEleL2_caloPhi"),
        Hist("TkEleL2_ptRatio", "TkEleL2_ptCorr/TkEleL2_originalPt"),
        Hist("GenZd_mass", "GenZd_mass"),
    ],

    "OSPair":[
        Hist("DPCandidates_pt_LeadEle", "DPCandidates_l1_pt"),
        Hist("DPCandidates_pt_SubLeadEle", "DPCandidates_l2_pt"),
        Hist("DPCandidates_dphi", "DPCandidates_dphi"),
        Hist("DPCandidates_deta", "DPCandidates_deta"),
        Hist("DPCandidates_dR", "DPCandidates_dR"),
        Hist("DPCandidates_dz", "DPCandidates_dz"),
        Hist("DPCandidates_score_LeadEle", "DPCandidates_l1_idScore"),
        Hist("DPCandidates_score_SubLeadEle", "DPCandidates_l2_idScore"),
        Hist("DPCandidates_mass", "DPCandidates_mass", rebin=100, ylim=(0,0.85)),
        Hist("DPCandidates_pt", "DPCandidates_pt"),
        Hist("DPCandidates_l1_ptRatio", "DPCandidates_l1_ptCorr/DPCandidates_l1_originalPt"),
        Hist("DPCandidates_l2_ptRatio", "DPCandidates_l2_ptCorr/DPCandidates_l2_originalPt"),
        #Hist2D("DPCandidates_mass:DPCandidates_l1_ptRatio", "DPCandidates_mass","DPCandidates_l1_ptCorr/DPCandidates_l1_originalPt"),
        #Hist2D("DPCandidates_mass:DPCandidates_l2_ptRatio", "DPCandidates_mass","DPCandidates_l2_ptCorr/DPCandidates_l2_originalPt"),
        Hist2D("DPCandidates_mass:DPCandidates_l1_pt", "DPCandidates_mass","DPCandidates_l1_pt"),
        Hist2D("DPCandidates_mass:DPCandidates_l2_pt", "DPCandidates_mass","DPCandidates_l2_pt"),
        #Hist2D("DPCandidates_mass:DPCandidates_l1_eta", "DPCandidates_mass","DPCandidates_l1_eta"),
        #Hist2D("DPCandidates_mass:DPCandidates_l2_eta", "DPCandidates_mass","DPCandidates_l2_eta"),
        #Hist2D("DPCandidates_mass:DPCandidates_deta", "DPCandidates_mass","DPCandidates_deta"),
        #Hist2D("DPCandidates_mass:DPCandidates_dphi", "DPCandidates_mass","DPCandidates_dphi"),
        #Hist2D("DPCandidates_mass:DPCandidates_dR", "DPCandidates_mass","DPCandidates_dR"),
        #Hist2D("DPCandidates_l1_pt:DPCandidates_l1_ptRatio", "DPCandidates_l1_pt", "DPCandidates_l1_ptCorr/DPCandidates_l1_originalPt"),
        #Hist2D("DPCandidates_l2_pt:DPCandidates_l2_ptRatio", "DPCandidates_l2_pt", "DPCandidates_l2_ptCorr/DPCandidates_l2_originalPt"),
    ],
}
