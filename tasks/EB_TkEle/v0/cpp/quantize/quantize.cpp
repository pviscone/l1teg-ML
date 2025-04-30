#pragma once
#include <alias.cpp>
#include <ap_fixed.h>


template <typename ap_t, typename T>
RVec<T> quantize(const RVec<T> &x){
    RVec<ap_t> res_q;
    RVec<T> res;
    std::transform(x.begin(), x.end(), std::back_inserter(res),
    [](T xi) -> ap_t { return (ap_t) xi; });

    std::transform(res.begin(), res.end(), std::back_inserter(res_q),
    [](ap_t xi) -> T { return (ap_t) xi; });

    return res;
}
