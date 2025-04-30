import ROOT
import os

def declare(filename, nbits):
    cpp_code = """
    #ifndef __CONIFER_H__
    #define __CONIFER_H__
    #include <ap_fixed.h>
    #include <conifer.h>
    #include <alias.cpp>

    typedef ap_fixed< <NBITS>, 1, AP_RND_CONV, AP_SAT> input_t;
    typedef ap_fixed<11, 4, AP_RND_CONV, AP_SAT> score_t;
    conifer::BDT< input_t, score_t , false> bdt(<FILENAME>);


    RVecF bdt_evaluate(const std::vector<std::variant<RVecF, RVecI, RVecD>> &input) {
        int n_features = input.size();
        int n_tkEle = std::visit([] (const auto& rvec){return rvec.size();}, input[0]);
        RVecF res(n_tkEle);
        for (int tkEle_idx = 0; tkEle_idx < n_tkEle; tkEle_idx++) {
            std::vector<float> x(n_features);
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
            res[tkEle_idx] = bdt._decision_function_float(x)[0];
        }
        return res/8;
    }
    #endif
    """
    #add include path
    this_dir = os.path.dirname(__file__)
    ROOT.gInterpreter.AddIncludePath(this_dir)
    ROOT.gInterpreter.AddIncludePath(os.path.join(this_dir, "conifer"))
    ROOT.gInterpreter.AddIncludePath(os.path.join(this_dir, "conifer/Vitis_HLS/simulation_headers/include"))

    cpp_code = cpp_code.replace("<FILENAME>", f'"{filename}"').replace("<NBITS>", str(nbits))
    ROOT.gInterpreter.Declare(cpp_code)