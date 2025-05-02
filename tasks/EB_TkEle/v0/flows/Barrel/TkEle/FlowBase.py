from cmgrdf_cli.flows import Tree
from CMGRDF import Define, Cut, ReDefine
from CMGRDF.collectionUtils import AliasCollection, DefineSkimmedCollection

from cpp import include_xilinx

include_xilinx.declare("quantize")

cryclu_name = "DecEmCaloBarrel"
tk_name = "DecTkBarrel"
genele_name = "GenEl"


def flow(region="EB"):
    if region == "EB":
        region_sel = " < 1.479"
    elif region == "EE":
        region_sel = " > 1.479"
    else:
        raise ValueError("Invalid region. Choose 'EB' or 'EE'.")

    tree = Tree()

    tree.add(
        "matching",
        [
            #! ---------- Acceptance selection --------- #
            DefineSkimmedCollection("GenEl", mask=f"abs(GenEl_eta) {region_sel}"),
            #! ----------------- Alias ----------------- #
            AliasCollection("GenEle", genele_name),
            AliasCollection("CryClu", cryclu_name),
            AliasCollection("Tk", tk_name),
            #! -------------- Cut ---------------------- #
            Cut("nGenEle_EB>0", "nGenEle > 0", samplePattern="(?!MinBias).*"),
            Cut("nGenEle==0", "nGenEle == 0", samplePattern="MinBias.*"),
            #! ------------- Matching Idxs ------------- #
            Define(
                "_match_idxs_nTkMatch_sumTkPt",
                "match_signal(GenEle_caloeta, GenEle_calophi, CryClu_eta, CryClu_phi, Tk_caloEta, Tk_caloPhi,Tk_pt)",
                samplePattern="(?!MinBias).*",
            ),
            Define(
                "_match_idxs_nTkMatch_sumTkPt",
                "match_bkg(CryClu_eta, CryClu_phi, Tk_caloEta, Tk_caloPhi,Tk_pt)",
                samplePattern="MinBias.*",
            ),
            Define("TkEle_GenEle_idx", "std::get<0>(_match_idxs_nTkMatch_sumTkPt)"),
            Define("TkEle_CryClu_idx", "std::get<1>(_match_idxs_nTkMatch_sumTkPt)"),
            Define("TkEle_Tk_idx", "std::get<2>(_match_idxs_nTkMatch_sumTkPt)"),
            Define("nTkEle", "TkEle_GenEle_idx.size()", plot="gen"),
            Cut("nTkEle>0", "nTkEle > 0"),
            #! -------------- Copy Features ----------- #
            DefineSkimmedCollection(
                "TkEle_GenEle",
                "GenEle",
                indices="TkEle_GenEle_idx",
                samplePattern="(?!MinBias).*",
                members=["pt", "eta", "phi", "caloeta", "calophi"],
            ),
            DefineSkimmedCollection(
                "TkEle_GenEle",
                "GenEle",
                define="RVecF(nTkEle, -1.)",
                samplePattern="MinBias.*",
            ),
            DefineSkimmedCollection(
                "TkEle_CryClu",
                "CryClu",
                indices="TkEle_CryClu_idx",
                members=["pt", "eta", "phi", "caloIso", "hwQual", "showerShape"],
            ),
            DefineSkimmedCollection(
                "TkEle_Tk",
                "Tk",
                indices="TkEle_Tk_idx",
                members=[
                    "pt",
                    "eta",
                    "phi",
                    "caloEta",
                    "caloPhi",
                    "chi2Bend",
                    "chi2RZ",
                    "chi2RPhi",
                    "nStubs",
                    "hitPattern",
                ],
            ),
            #! -------------- New Features ------------ #
            Define(
                "TkEle_absdeta", "round(abs(TkEle_CryClu_eta - TkEle_Tk_caloEta)*720/M_PI)"
            ),
            Define(
                "TkEle_absdphi",
                "round(abs(DeltaPhi(TkEle_CryClu_phi, TkEle_Tk_caloPhi))*720/M_PI)",
            ),
            Define("TkEle_CryClu_standaloneWP", "(TkEle_CryClu_hwQual & 0x1);"),
            Define(
                "TkEle_CryClu_looseL1TkMatchWP", "(TkEle_CryClu_hwQual & 0x2)==0x2;"
            ),
            Define("TkEle_label", "RVecI(nTkEle,1)", samplePattern="(?!MinBias).*"),
            Define("TkEle_label", "RVecI(nTkEle,0)", samplePattern="MinBias.*"),
            Define("TkEle_nTkMatch", "std::get<3>(_match_idxs_nTkMatch_sumTkPt)"),
            Define("TkEle_sumTkPt", "std::get<4>(_match_idxs_nTkMatch_sumTkPt)"),
            #! -------------- Clu Ratios --------------------- #
            Define("TkEle_CryClu_relIso", "TkEle_CryClu_caloIso / TkEle_CryClu_pt"),
            #! -------------- Tk Ratios --------------------- #
            Define("TkEle_PtRatio", "TkEle_CryClu_pt/TkEle_Tk_pt"),
            Define("TkEle_Tk_ptFrac", "TkEle_sumTkPt/TkEle_Tk_pt"),
            #! -------------- Cluster ratio quantization ------------ #
            ReDefine(
                "TkEle_CryClu_showerShape",
                "quantize<ap_ufixed<6, 0, AP_RND_CONV, AP_SAT>>(TkEle_CryClu_showerShape)",
            ),
            ReDefine(
                "TkEle_CryClu_relIso",
                "quantize<ap_ufixed<6, 0, AP_RND_CONV, AP_SAT>>(TkEle_CryClu_relIso/16.)",
            ),
            #! -------------- Tk ratio binning ------------ #
            ReDefine(
                "TkEle_Tk_chi2RPhi",
                "get_hist_idx(TkEle_Tk_chi2RPhi, RVecF({0.0, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 10.0, 15.0, 20.0, 35.0, 60.0, 200.0}))",
            ),
        ],
    )
    return tree
