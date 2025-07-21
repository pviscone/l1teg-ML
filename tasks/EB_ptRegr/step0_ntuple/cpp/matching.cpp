#pragma once
#include "ROOT/RVec.hxx"
#include "Math/Vector4Dfwd.h"

using namespace ROOT::VecOps;
using namespace ROOT::Math;
using namespace ROOT;



RVec<int> match(
        const RVecF &obj1_eta,
        const RVecF &obj1_phi,
        const RVecF &obj2_eta,
        const RVecF &obj2_phi,
        float dRcut = 0.1){

    RVec<int> idx_v(obj1_eta.size(), -1);
    int nObj2 = obj2_eta.size();

    for(int i2 = 0; i2 < nObj2; i2++) {
        RVecF dR=DeltaR(obj1_eta, RVecF(obj1_eta.size(),obj2_eta[i2]), obj1_phi, RVecF(obj1_eta.size(),obj2_phi[i2]));
        for (int i = 0; i < obj1_eta.size(); i++) {
            if (dR[i] < dRcut) {
                idx_v[i] = i2;
            }
        }
    }
    return idx_v;
}


RVec<int> count_matched(
    const RVecI &genidx,
    const int &ngen
){
    RVec<int> count(ngen, 0);
    for (int i=0; i<ngen; i++) {
        count[i] = Sum(genidx==i);
    }
    return count;
}