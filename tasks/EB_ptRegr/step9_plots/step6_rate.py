#%%
import sys
import os
os.makedirs("plots/step6_rate", exist_ok=True)
sys.path.append("../../../cmgrdf-cli/cmgrdf_cli/plots")

from plotters import TRate
from common import minbias
import uproot as up
import hist
import ROOT


ROOT.EnableImplicitMT()


def declare(filename, in_q, out_q):
    cpp_code = """
    using namespace ROOT;
    using namespace ROOT::VecOps;
    #ifndef __CONIFER_H__
    #define __CONIFER_H__
    #include <ap_fixed.h>
    #include <conifer.h>

    float init_pred = 512. * pow(2, -9);

    typedef ap_fixed< <IN_Q_0>, <IN_Q_1>, AP_RND_CONV, AP_SAT> input_t;
    typedef ap_fixed< <OUT_Q_0>, <OUT_Q_1>, AP_RND_CONV, AP_SAT> score_t;
    conifer::BDT< input_t, score_t , false> bdt(<FILENAME>);


    RVecF bdt_evaluate(const std::vector<std::variant<RVecF, RVecI, RVecD>> &input) {
        int n_features = input.size();
        int n_tkEle = std::visit([] (const auto& rvec){return rvec.size();}, input[0]);
        RVecF res(n_tkEle);
        for (int tkEle_idx = 0; tkEle_idx < n_tkEle; tkEle_idx++) {
            std::vector<float> x(n_features);
            std::vector<input_t> xt(n_features);
            for (int feat_idx = 0; feat_idx < n_features; feat_idx++) {
                x[feat_idx] = std::visit([tkEle_idx] (const auto& rvec){
                    using T = std::decay_t<decltype(rvec)>;
                    if constexpr (std::is_same_v<T, RVecF>) {
                        return rvec[tkEle_idx];
                    } else {
                        return static_cast<float>(rvec[tkEle_idx]);
                    }
                }, input[feat_idx]);
                std::transform(x.begin(), x.end(), std::back_inserter(xt),
                   [](float xi) -> input_t { return (input_t) xi; });
            }
            res[tkEle_idx] = init_pred+(bdt.decision_function(xt)[0]).to_float();
        }
        return res;
    }
    #endif
    """
    #add include path
    this_dir = os.path.dirname(__file__)
    ROOT.gInterpreter.AddIncludePath(this_dir)
    ROOT.gInterpreter.AddIncludePath(os.path.join(this_dir, "../../../utils/conifer"))
    ROOT.gInterpreter.AddIncludePath(os.path.join(this_dir, "../../../utils/conifer/conifer/backends/cpp/include"))
    ROOT.gInterpreter.AddIncludePath(os.path.join(this_dir, "../../../utils/conifer/conifer/externals"))
    ROOT.gInterpreter.AddIncludePath(os.path.join(this_dir, "../../../utils/conifer/conifer/externals/Vitis_HLS/simulation_headers/include"))
    ROOT.gInterpreter.AddIncludePath(os.path.join(this_dir, "../../../utils/conifer/conifer/externals/nlohmann/include"))

    cpp_code = cpp_code.replace("<FILENAME>", f'"{filename}"').replace("<IN_Q_0>", str(in_q[0])).replace("<IN_Q_1>", str(in_q[1])).replace("<OUT_Q_0>", str(out_q[0])).replace("<OUT_Q_1>", str(out_q[1]))
    ROOT.gInterpreter.Declare(cpp_code)


df = ROOT.RDataFrame("Events", minbias, ["TkEleL2_pt", "TkEleL2_ptCorr", "TkEleL2_hwQual", "TkEleL2_caloEta"])

tot_events = df.Count().GetValue()

declare("models/conifer_model_L1_q10_hls.json", (10, 1), (12, 3))
df = (df.Define("mask", "abs(TkEleL2_caloEta)<1.479")
    .Redefine("TkEleL2_pt", "TkEleL2_pt[mask]")
    .Redefine("TkEleL2_ptCorr", "TkEleL2_ptCorr[mask]")
    .Redefine("TkEleL2_hwQual", "TkEleL2_hwQual[mask]")
    .Redefine("TkEleL2_caloEta", "TkEleL2_caloEta[mask]")
    .Filter("TkEleL2_pt.size()>0")
    .Define("Lead_pt", "TkEleL2_pt[0]")
    .Redefine("TkEleL2_ptCorr", "ROOT::VecOps::Reverse(ROOT::VecOps::Sort(TkEleL2_ptCorr))")
    .Define("rescaled_hwCaloEta", "-1 + abs(TkEleL2_caloEta)/pow(2,8)")
    .Define("rescaled_caloTkAbsDphi", "-1 + TkEleL2_in_caloTkAbsDphi/pow(2,5)")
    .Define("rescaled_hwTkChi2RPhi", "-1 + TkEleL2_in_hwTkChi2RPhi/pow(2,3)")
    .Define("rescaled_caloPt", "-1 + (TkEleL2_in_caloPt - 1)/pow(2,6)")
    .Define("rescaled_caloSS", "-1 + TkEleL2_in_caloSS*2")
    .Define("rescaled_caloTkPtRatio", "-1 + TkEleL2_in_caloTkPtRatio/pow(2,3)")
    .Define("Lead_ptCorr", "TkEleL2_pt * bdt_evaluate({rescaled_hwCaloEta, rescaled_caloTkAbsDphi, rescaled_hwTkChi2RPhi, rescaled_caloPt, rescaled_caloSS, rescaled_caloTkPtRatio})")
    .Define("tight_mask", "(TkEleL2_hwQual & 2) == 2")
)
#nevents = df.Count().GetValue()


df_tight = (
    df.Redefine("TkEleL2_pt", "TkEleL2_pt[tight_mask]")
    .Redefine("TkEleL2_ptCorr", "TkEleL2_ptCorr[tight_mask]")
    .Filter("TkEleL2_pt.size()>0")
    .Define("LeadTight_pt", "TkEleL2_pt[0]")
    .Redefine("TkEleL2_ptCorr", "ROOT::VecOps::Reverse(ROOT::VecOps::Sort(TkEleL2_ptCorr))")
    .Define("LeadTight_ptCorr", "TkEleL2_ptCorr[0]")
)
nevents_tight = df_tight.Count().GetValue()

res = df.AsNumpy(["Lead_pt", "Lead_ptCorr"])
res_tight = df_tight.AsNumpy(["LeadTight_pt", "LeadTight_ptCorr"])

#%%
def fill(array, nev):
    axis = hist.axis.Regular(100, 0, 100)
    h = hist.Hist(axis, storage=hist.storage.Weight())
    h.fill(array)
    h = h/h.integrate(0).value
    h = h * 31000 * nev/tot_events
    return h

def ptcorr_scaling(x):
    return 1.07*x+1.26

def pt_scaling(x):
    return 1.09*x+2.82


h_pt_tight_on = fill(res_tight["LeadTight_pt"], nevents_tight)
h_pt_corr_tight_on = fill(res_tight["LeadTight_ptCorr"], nevents_tight)
h_pt_tight_off = fill(pt_scaling(res_tight["LeadTight_pt"]), nevents_tight)
h_pt_corr_tight_off = fill(ptcorr_scaling(res_tight["LeadTight_ptCorr"]), nevents_tight)


#%%
rate = TRate(ylim=(1, 4e4), xlim=(-1,80), xlabel = "Offline $p_{T}$ [GeV]", cmstext="Phase-2 Simulation Preliminary", lumitext="PU 200")


rate.add(h_pt_corr_tight_off, label="Regressed (Tight)")
rate.add(h_pt_tight_off, label="No Regression (Tight)")
rate.ax.axhline(18, color="gray", linestyle="--", alpha=0.5)
rate.save("plots/step6_rate/step6_rate_rate.pdf")
rate.save("plots/step6_rate/step6_rate_rate.png")

#%%
rate = TRate(ylim=(1, 4e4), xlim=(-1,80), xlabel = "Online $p_{T}$ [GeV]", cmstext="Phase-2 Simulation Preliminary", lumitext="PU 200")


rate.add(h_pt_corr_tight_on, label="Regressed (Tight)")
rate.add(h_pt_tight_on, label="No Regression (Tight)")
rate.ax.axhline(18, color="gray", linestyle="--", alpha=0.5)
rate.save("plots/step6_rate/step6_rate_rate.pdf")
rate.save("plots/step6_rate/step6_rate_rate.png")

# %%
