from cmgrdf_cli.flows import Tree
from CMGRDF import Define, Cut, ReDefine
from CMGRDF.collectionUtils import DefineSkimmedCollection
import ROOT
import os
import sys

sys.path.append("../")
sys.path.append("../../../utils/conifer")
from common import quant, q_out, features_q, conifermodel, init_pred

in_q = (quant, 1)
out_q = (q_out[0], q_out[1])
def declare(filename, in_q, out_q):
    cpp_code = """
    using namespace ROOT;
    using namespace ROOT::VecOps;
    #ifndef __CONIFER_H__
    #define __CONIFER_H__
    #include <ap_fixed.h>
    #include <conifer.h>


    typedef ap_fixed< <IN_Q_0>, <IN_Q_1>, AP_RND_CONV, AP_SAT> input_t;
    typedef ap_fixed< <OUT_Q_0>, <OUT_Q_1>, AP_RND_CONV, AP_SAT> score_t;
    conifer::BDT< input_t, score_t , false> bdt(<FILENAME>);


    RVecF bdt_evaluate(const std::vector<std::variant<RVecF, RVecI, RVecD>> &input, bool debug=false) {
        int n_features = input.size();
        int n_tkEle = std::visit([] (const auto& rvec){return rvec.size();}, input[0]);
        RVecF res(n_tkEle);
        for (int tkEle_idx = 0; tkEle_idx < n_tkEle; tkEle_idx++) {
            std::vector<float> x(n_features);
            std::vector<input_t> xt;
            xt.reserve(n_features);
            for (int feat_idx = 0; feat_idx < n_features; feat_idx++) {
                x[feat_idx] = std::visit([tkEle_idx] (const auto& rvec){
                    using T = std::decay_t<decltype(rvec)>;
                    if constexpr (std::is_same_v<T, RVecF>) {
                        return rvec[tkEle_idx];
                    } else {
                        return static_cast<float>(rvec[tkEle_idx]);
                    }
                }, input[feat_idx]);
            }
            std::transform(x.begin(), x.end(), std::back_inserter(xt),
                   [](float xi) -> input_t { return (input_t) xi; });
            if(debug) {
                for (int feat_idx = 0; feat_idx < n_features; feat_idx++) {
                    std::cout << "Feature " << feat_idx << ": " << x[feat_idx] << " -> " << xt[feat_idx] << std::endl;
                }
            }
            res[tkEle_idx] = (bdt.decision_function(xt)[0]).to_float();
            if(debug) {
                std::cout << "Decision function output for tkEle " << tkEle_idx << ": " << res[tkEle_idx] << std::endl << std::endl << std::endl;
            }
        }
        return res;
    }
    #endif
    """
    #add include path
    this_dir = os.path.dirname(__file__)
    ROOT.gInterpreter.AddIncludePath(this_dir)
    ROOT.gInterpreter.AddIncludePath(os.path.join(this_dir, "../../../../utils/conifer"))
    ROOT.gInterpreter.AddIncludePath(os.path.join(this_dir, "../../../../utils/conifer/conifer/backends/cpp/include"))
    ROOT.gInterpreter.AddIncludePath(os.path.join(this_dir, "../../../../utils/conifer/conifer/externals"))
    ROOT.gInterpreter.AddIncludePath(os.path.join(this_dir, "../../../../utils/conifer/conifer/externals/Vitis_HLS/simulation_headers/include"))
    ROOT.gInterpreter.AddIncludePath(os.path.join(this_dir, "../../../../utils/conifer/conifer/externals/nlohmann/include"))

    filename = os.path.abspath(filename)

    cpp_code = cpp_code.replace("<FILENAME>", f'"{filename}"').replace("<IN_Q_0>", str(in_q[0])).replace("<IN_Q_1>", str(in_q[1])).replace("<OUT_Q_0>", str(out_q[0])).replace("<OUT_Q_1>", str(out_q[1]))
    ROOT.gInterpreter.Declare(cpp_code)

declare(conifermodel, in_q, out_q)
def flow():
    tree = Tree()
    tree.add("noRegress", [
                      DefineSkimmedCollection("GenEl", mask="GenEl_prompt==2"),
                      Cut("nGenElBarrel==2", "Sum(abs(GenEl_caloeta)<1.479)==2", samplePattern="(?!MinBias).*"),
                      Define("GenEl_p4", "makeP4(GenEl_pt, GenEl_eta,GenEl_phi,0.000511)"),
                      Define("GenZd_p4", "GenEl_p4[0]+GenEl_p4[1]"),
                      Define("GenZd_mass", "GenZd_p4.mass()"),
                      Define("TkEleL2_originalPt", "TkEleL2_pt")])

    #chi2rphi_bins = "RVecF({0.0, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 10.0, 15.0, 20.0, 35.0, 60.0, 200.0})"
    tree.add("regressed", [

        Define("TkEleL2_rescaled_idScore", "TkEleL2_idScore"),
        Define("TkEleL2_rescaled_caloEta", "-1 + abs(TkEleL2_caloEta)"),
        Define("TkEleL2_rescaled_caloTkAbsDphi", "-1 + TkEleL2_in_caloTkAbsDphi/pow(2,5)"),
        Define("TkEleL2_rescaled_hwTkChi2RPhi", "-1 + TkEleL2_in_hwTkChi2RPhi/pow(2,3)"), #TODO check
        Define("TkEleL2_rescaled_caloPt", "-1 + (TkEleL2_in_caloPt)/pow(2,5)"),
        Define("TkEleL2_rescaled_caloSS", "-1 + TkEleL2_in_caloSS*2"),
        Define("TkEleL2_rescaled_caloTkPtRatio", "-1 + TkEleL2_in_caloTkPtRatio/pow(2,3)"),
        Define("out", f"( {init_pred} + bdt_evaluate({{ {','.join([f'TkEleL2_rescaled_{f}' for f in features_q])} }}))"),
        ReDefine("TkEleL2_pt", "TkEleL2_pt * out"),
    ], parent=["noRegress"])


    tree.add("main_{leaf}",[

        #!------------------ TkEle ------------------!#
        DefineSkimmedCollection("TkEleL2", mask="abs(TkEleL2_caloEta)<1.479 && TkEleL2_in_caloPt>0"),
        Define("TkEleMask", "TkEleL2_idScore>-0.3 && TkEleL2_pt>4", plot="noSelection"),
        DefineSkimmedCollection("TkEleL2", mask="TkEleMask"),
        Cut("2tkEle","nTkEleL2>=2", plot="2tkEle_pt3_score_m0p3"),

        #!------------------ OS ------------------!#
        Define("DPCandidatesIdxs", "makeDPCandidateOSIdx(TkEleL2_charge)"),
        Cut("1OS pair", "DPCandidatesIdxs.first.size()>0"),
        DefineSkimmedCollection("DPCandidates_l1", "TkEleL2", indices="DPCandidatesIdxs.first"),
        DefineSkimmedCollection("DPCandidates_l2", "TkEleL2", indices="DPCandidatesIdxs.second"),
        Define("DPCandidates_l1_p4", "makeP4(DPCandidates_l1_pt, DPCandidates_l1_eta, DPCandidates_l1_phi, 0.0005)"),
        Define("DPCandidates_l2_p4", "makeP4(DPCandidates_l2_pt, DPCandidates_l2_eta, DPCandidates_l2_phi, 0.0005)"),
        Define("DPCandidates_p4", "DPCandidates_l1_p4+DPCandidates_l2_p4"),
        Define("DPCandidates_mass", "getMasses(DPCandidates_p4)"),
        Define("DPCandidates_pt", "getPts(DPCandidates_p4)"),
        Define("DPCandidates_dz", "abs(DPCandidates_l1_vz-DPCandidates_l2_vz)"),
        Define("DPCandidates_dphi", "abs(ROOT::VecOps::DeltaPhi(DPCandidates_l1_phi, DPCandidates_l2_phi))"),
        Define("DPCandidates_deta", "abs(DPCandidates_l1_eta-DPCandidates_l2_eta)"),
        Define("DPCandidates_dR", "ROOT::VecOps::sqrt(DPCandidates_deta*DPCandidates_deta+DPCandidates_dphi*DPCandidates_dphi)", plot="OSPair"),
        DefineSkimmedCollection("DPCandidates", mask="DPCandidates_dz<0.65"),
        Cut(">=1 DPCandidate (dz<0.65)", "nDPCandidates>0", plot="dZCut"),

        #!------------------ Gen Matching ------------------!#
        DefineSkimmedCollection("DPCandidates", mask="""
                (match_mask(DPCandidates_l1_caloEta, DPCandidates_l1_caloPhi, GenEl_caloeta[0], GenEl_calophi[0]) &&
                match_mask(DPCandidates_l2_caloEta, DPCandidates_l2_caloPhi, GenEl_caloeta[1], GenEl_calophi[1])) ||
                (match_mask(DPCandidates_l1_caloEta, DPCandidates_l1_caloPhi, GenEl_caloeta[1], GenEl_calophi[1]) &&
                match_mask(DPCandidates_l2_caloEta, DPCandidates_l2_caloPhi, GenEl_caloeta[0], GenEl_calophi[0]))
        """),
        Cut(">=1 DPCandidate (matched to GenEl)", "nDPCandidates>0", plot="matchingGenCut"),

    ], parent=["regressed", "noRegress"])
    return tree
