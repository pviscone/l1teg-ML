#ifndef  _XEEFUNCTIONS_H
#define  _XEEFUNCTIONS_H
#include "ROOT/RVec.hxx"
#include "Math/Vector4Dfwd.h"

using namespace ROOT;

std::pair<RVecI, RVecI> makeDPCandidateOSIdx(const RVecI& charge) {
    size_t n = charge.size();
    RVecI res1;
    RVecI res2;
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            if (charge[i] * charge[j] < 0) {
                res1.push_back(i);
                res2.push_back(j);
            }
        }
    }
    return { res1, res2 };
}
#endif


template <typename T, typename U>
RVec<U> apply(const T& x, std::function<U(const T&)> func) {
    RVec<U> res(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        res[i] = func(x[i]);
    }
    return res;
}

RVecF getMasses(const RVec<ROOT::Math::PtEtaPhiMVector>& vec) {
    RVecF masses(vec.size());
    for (size_t i = 0; i < vec.size(); ++i) {
        masses[i] = vec[i].mass();
    }
    return masses;
}

RVecF getPts(const RVec<ROOT::Math::PtEtaPhiMVector>& vec) {
    RVecF pts(vec.size());
    for (size_t i = 0; i < vec.size(); ++i) {
        pts[i] = vec[i].pt();
    }
    return pts;
}


template<typename T>
RVec<bool> trimMask(const RVec<T>& vec, size_t n) {
    RVec<bool> mask(vec.size(), true);
    if (n >= vec.size()) {
        return mask;
    }
    for (size_t i = n; i < vec.size(); ++i) {
        mask[i] = false;
    }
    return vec;
}
