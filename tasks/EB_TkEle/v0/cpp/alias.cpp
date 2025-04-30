#pragma once
#include "ROOT/RVec.hxx"

typedef ROOT::RVecF RVecF;
typedef ROOT::RVecI RVecI;
typedef ROOT::RVecD RVecD;

template <typename T> using RVec = ROOT::VecOps::RVec<T>;

namespace VecOps = ROOT::VecOps;