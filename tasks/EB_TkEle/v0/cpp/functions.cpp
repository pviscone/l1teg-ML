#pragma once
#include "alias.cpp"

auto generator = TRandom();

//_____________________________________________________________________________

float if3(bool cond, float iftrue, float iffalse) {
    return cond ? iftrue : iffalse;
}

// Generic Take: recursively handles nested RVecs.
template <typename InContainer, typename IdxContainer>
InContainer GenericTake(const InContainer& in, const IdxContainer& indexes) {
    // Determine the type of an element of the indexes container.
    using IndexElem = std::decay_t<decltype(indexes[0])>;

    if constexpr (std::is_integral_v<IndexElem>) {
        return ROOT::VecOps::Take(in, indexes);
    } else {
        // Recursive case: indexes are containers (e.g. RVec<int>, etc.).
        // Recursively call Take on each corresponding sub-container.
        using SubType = std::decay_t<decltype(in[0])>;
        RVec<SubType> result;
        result.reserve(indexes.size());
        for (size_t i = 0; i < indexes.size(); ++i) {
            result.push_back(GenericTake(in[i], indexes[i]));
        }
        return result;
    }
}

template <typename T>
RVec<T> operator<<(const RVec<T>& vec, int shift) {
    RVec<T> result;
    result.reserve(vec.size());
    for (const auto& v : vec) {
        result.push_back(v*pow(2,shift));
    }
    return result;
}

template <typename T>
RVec<T> operator>>(const RVec<T>& vec, int shift) {
    RVec<T> result;
    result.reserve(vec.size());
    for (const auto& v : vec) {
        result.push_back(v*pow(2,-shift));
    }
    return result;
}

RVec<int> operator<<(const RVec<int>& vec, int shift) {
    RVec<int> result;
    result.reserve(vec.size());
    for (const auto& v : vec) {
        result.push_back(v << shift);
    }
    return result;
}

RVec<int> operator>>(const RVec<int>& vec, int shift) {
    RVec<int> result;
    result.reserve(vec.size());
    for (const auto& v : vec) {
        result.push_back(v >> shift);
    }
    return result;
}


template<typename T, typename U>
RVec<T> bitscale(const RVec<T>& x, U inf, U min_x, int bitshift) {
    return inf + (x - min_x)/pow(2,bitshift);
}

template <typename T>
RVec<bool> maskMaxPerGroup(const RVec<T> &scores, const RVecI &idxs) {
  std::unordered_map<int, T> groupMax;
  // Compute max scores for each group.
  for (size_t i = 0; i < scores.size(); ++i) {
    if (groupMax.find(idxs[i]) == groupMax.end() || //If the group is not in the map
        scores[i] > groupMax[idxs[i]])              //If the score is greater than the max score of the group
      groupMax[idxs[i]] = scores[i];                //Update the max score of the group
  }

  RVec<bool> mask(scores.size(), false);
  // Set true at positions corresponding to group max.
  for (size_t i = 0; i < scores.size(); ++i) {
    if (scores[i] == groupMax[idxs[i]])
      mask[i] = true;
  }
  return mask;
}

template <typename T, typename U>
RVecI get_hist_idx(const RVec<T> &x, const RVec<U> &low_bin_edges){
    RVecI res =VecOps::Map(x, [&low_bin_edges] (const T &value){return VecOps::ArgMax(low_bin_edges[low_bin_edges<=value]);});
    return res;
}