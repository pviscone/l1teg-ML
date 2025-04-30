#pragma once
#include "alias.cpp"


//_____________________________________________________________________________

RVecF dR(
        const RVecF &eta1,
        const float &eta2,
        const RVecF &phi1,
        const float &phi2){

    RVecF dR = ROOT::VecOps::sqrt(ROOT::VecOps::pow(eta1 - eta2, 2) +
            ROOT::VecOps::pow(ROOT::VecOps::DeltaPhi(phi1, phi2), 2));
    return dR;
}

RVec<bool> match_mask(
        const RVecF &obj1_eta,
        const RVecF &obj1_phi,
        const float &obj2_eta,
        const float &obj2_phi,
        float dRcut = 0.1){

    RVec<bool> mask(obj1_eta.size(), false);
    RVecF dR=ROOT::VecOps::DeltaR(obj1_eta, RVecF(obj1_eta.size(),obj2_eta), obj1_phi, RVecF(obj1_eta.size(),obj2_phi));

    for (int i = 0; i < obj1_eta.size(); i++) {
        if (dR[i] < dRcut) {
            mask[i] = true;
        }
    }
    return mask;
}

std::tuple<RVecI,RVecI,RVecI,RVecI,RVecF> match_signal(
                                        RVecF genEta,
                                        RVecF genPhi,
                                        RVecF caloEta,
                                        RVecF caloPhi,
                                        RVecF tkEta,
                                        RVecF tkPhi,
                                        RVecF tkPt,
                                        float dphi_cut = 0.3,
                                        float deta_cut = 0.03,
                                        float dRgen_cut = 0.1
                                    ){
    RVecI genIdxs;
    RVecI caloIdxs;
    RVecI tkIdxs;

    RVecI nTkMatch;
    RVecF sum_TkPt;

    int ngen = genEta.size();
    int ncalo = caloEta.size();
    int ntk = tkEta.size();
    for (int gen_idx = 0; gen_idx < ngen; gen_idx++) {
        for (int calo_idx = 0; calo_idx < ncalo; calo_idx++) {
            int matched_tk=0;
            float sum_tkpt=0;
            if (ROOT::VecOps::DeltaR(genEta[gen_idx], caloEta[calo_idx], genPhi[gen_idx], caloPhi[calo_idx]) > dRgen_cut) {continue;}
            for (int tk_idx=0; tk_idx < ntk; tk_idx++){
                if (pow(abs(caloEta[calo_idx] - tkEta[tk_idx])/deta_cut,2) + pow(ROOT::VecOps::DeltaPhi(caloPhi[calo_idx], tkPhi[tk_idx])/dphi_cut,2)>1) {continue;}
                matched_tk++;
                sum_tkpt+=tkPt[tk_idx];
                genIdxs.push_back(gen_idx);
                caloIdxs.push_back(calo_idx);
                tkIdxs.push_back(tk_idx);
            }
            for(int j=0; j<matched_tk; j++){
                nTkMatch.push_back(matched_tk);
                sum_TkPt.push_back(sum_tkpt);
            }
        }
    }
    return {genIdxs, caloIdxs, tkIdxs, nTkMatch, sum_TkPt};
}



std::tuple<RVecI,RVecI,RVecI,RVecI,RVecF> match_bkg(
                                    RVecF caloEta,
                                    RVecF caloPhi,
                                    RVecF tkEta,
                                    RVecF tkPhi,
                                    RVecF tkPt,
                                    float dphi_cut = 0.3,
                                    float deta_cut = 0.03
                                    ){
    RVecI genIdxs;
    RVecI caloIdxs;
    RVecI tkIdxs;

    RVecI nTkMatch;
    RVecF sum_TkPt;

    int ncalo = caloEta.size();
    int ntk = tkEta.size();

    for (int calo_idx = 0; calo_idx < ncalo; calo_idx++) {
        int matched_tk=0;
        float sum_tkpt=0;
        for (int tk_idx=0; tk_idx < ntk; tk_idx++){
            if (pow(abs(caloEta[calo_idx] - tkEta[tk_idx])/deta_cut,2) + pow(ROOT::VecOps::DeltaPhi(caloPhi[calo_idx], tkPhi[tk_idx])/dphi_cut,2)>1) {continue;}
            matched_tk++;
            sum_tkpt+=tkPt[tk_idx];
            genIdxs.push_back(-1);
            caloIdxs.push_back(calo_idx);
            tkIdxs.push_back(tk_idx);
        }
        for(int j=0; j<matched_tk; j++){
            nTkMatch.push_back(matched_tk);
            sum_TkPt.push_back(sum_tkpt);
        }
    }

    return {genIdxs, caloIdxs, tkIdxs, nTkMatch, sum_TkPt};
}

RVecI get_dRmatchIdx( RVecF eta1, RVecF phi1, RVecF eta2, RVecF phi2, float dRcut = 0.1){
    RVecI res(eta1.size(),-1);
    for (int i = 0; i < eta1.size(); i++) {
        RVecF eta1_v(eta2.size(), eta1[i]);
        RVecF phi1_v(phi2.size(), phi1[i]);
        RVecF dR = ROOT::VecOps::DeltaR(eta1_v, eta2, phi1_v, phi2);
        for (int j = 0; j < eta2.size(); j++) {
            if (dR[j] < dRcut) {
                res[i] = j;
            }
        }
    }
    return res;
}