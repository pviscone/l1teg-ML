#ifndef CONIFER_CPP_H__
#define CONIFER_CPP_H__
#include "nlohmann/include/nlohmann/json.hpp"
#include <cassert>
#include <fstream>

namespace conifer{

/* ---
* Balanced tree reduce implementation.
* Reduces an array of inputs to a single value using the template binary operator 'Op',
* for example summing all elements with Op_add, or finding the maximum with Op_max
* Use only when the input array is fully unrolled. Or, slice out a fully unrolled section
* before applying and accumulate the result over the rolled dimension.
* Required for emulation to guarantee equality of ordering.
* --- */
constexpr int floorlog2(int x) { return (x < 2) ? 0 : 1 + floorlog2(x / 2); }

template <int B>
constexpr int pow(int x) {
  return x == 0 ? 1 : B * pow<B>(x - 1);
}

constexpr int pow2(int x) { return pow<2>(x); }

template <class T, class Op>
T reduce(std::vector<T> x, Op op) {
  int N = x.size();
  int leftN = pow2(floorlog2(N - 1)) > 0 ? pow2(floorlog2(N - 1)) : 0;
  //static constexpr int rightN = N - leftN > 0 ? N - leftN : 0;
  if (N == 1) {
    return x.at(0);
  } else if (N == 2) {
    return op(x.at(0), x.at(1));
  } else {
    std::vector<T> left(x.begin(), x.begin() + leftN);
    std::vector<T> right(x.begin() + leftN, x.end());
    return op(reduce<T, Op>(left, op), reduce<T, Op>(right, op));
  }
}

template<class T>
class OpAdd {
public:
  T operator()(T a, T b) { return a + b; }
};

template <typename T>
std::function<bool (T, T)> createSplit(const std::string& op) {
    std::function<bool (T, T)> split;
    if (op == "<") {
        split = [](const T& a, const T& b) { return a < b; };
    } else if (op == "<=") {
        split = [](const T& a, const T& b) { return a <= b; };
    } else {
        throw std::invalid_argument("Invalid operator string: " + op);
    }
    return split;
}

template<class T, class U>
class DecisionTree{

private:
  std::vector<int> feature;
  std::vector<int> children_left;
  std::vector<int> children_right;
  std::vector<T> threshold_;
  std::vector<U> value_;
  std::vector<double> threshold;
  std::vector<double> value;
  std::function<bool (T, T)> split;

public:

  U decision_function(const std::vector<T> &x) const{
    /* Do the prediction */
    int i = 0;
    bool comparison;
    while(feature[i] != -2){ // continue until reaching leaf
      comparison = split(x[feature[i]], threshold_[i]);
      i = comparison ? children_left[i] : children_right[i];
    }
    return value_[i];
  }

  void init_(std::function<bool (T, T)> split){
    /* Since T, U types may not be readable from the JSON, read them to double and the cast them here */
    this->split = split;
    std::transform(threshold.begin(), threshold.end(), std::back_inserter(threshold_),
                   [](double t) -> T { return (T) t; });
    std::transform(value.begin(), value.end(), std::back_inserter(value_),
                   [](double v) -> U { return (U) v; });
  }

  // Define how to read this class to/from JSON
  NLOHMANN_DEFINE_TYPE_INTRUSIVE(DecisionTree, feature, children_left, children_right, threshold, value);

}; // class DecisionTree

template<class T, class U, bool useAddTree = false>
class BDT{

private:

  unsigned int n_classes;
  unsigned int n_trees;
  unsigned int n_features;
  double norm;
  std::vector<double> init_predict;
  std::vector<U> init_predict_;
  // vector of decision trees: outer dimension tree, inner dimension class
  std::vector<std::vector<DecisionTree<T,U>>> trees;
  OpAdd<U> add;

public:

  // Define how to read this class to/from JSON
  NLOHMANN_DEFINE_TYPE_INTRUSIVE(BDT, n_classes, n_trees, n_features, norm, init_predict, trees);

  BDT(std::string filename){
    /* Construct the BDT from conifer cpp backend JSON file */
    std::ifstream ifs(filename);
    nlohmann::json j = nlohmann::json::parse(ifs);
    from_json(j, *this);
    auto splitting_convention = j.value("splitting_convention", "<="); // read the splitting convention with default value of "<=" if it's unspecified
    auto split = createSplit<T>(splitting_convention);
    /* Do some transformation to initialise things into the proper emulation T, U types */
    if(n_classes == 2) n_classes = 1;
    std::transform(init_predict.begin(), init_predict.end(), std::back_inserter(init_predict_),
                   [](double ip) -> U { return (U) ip; });
    for(unsigned int i = 0; i < n_trees; i++){
      for(unsigned int j = 0; j < n_classes; j++){
        trees.at(i).at(j).init_(split);
      }
    }
  }

  std::vector<U> decision_function(std::vector<T> x) const{
    /* Do the prediction */
    assert("Size of feature vector mismatches expected n_features" && x.size() == n_features);
    std::vector<U> values;
    std::vector<std::vector<U>> values_trees;
    values_trees.resize(n_classes);
    values.resize(n_classes, U(0));
    for(unsigned int i = 0; i < n_classes; i++){
      std::transform(trees.begin(), trees.end(), std::back_inserter(values_trees.at(i)),
                     [&i, &x](auto tree_v){ return tree_v.at(i).decision_function(x); });
      if(useAddTree){
        values.at(i) = init_predict_.at(i);
        values.at(i) += reduce<U, OpAdd<U>>(values_trees.at(i), add);
      }else{
        values.at(i) = std::accumulate(values_trees.at(i).begin(), values_trees.at(i).end(), U(init_predict_.at(i)));
      }
      values.at(i) *= (U) norm;
    }

    return values;
  }

  std::vector<double> _decision_function_double(std::vector<double> x) const{
    /* Do the prediction with data in/out as double, cast to T, U before prediction */
    std::vector<T> xt;
    std::transform(x.begin(), x.end(), std::back_inserter(xt),
                   [](double xi) -> T { return (T) xi; });
    std::vector<U> y = decision_function(xt);
    std::vector<double> yd;
    std::transform(y.begin(), y.end(), std::back_inserter(yd),
                [](U yi) -> double { return (double) yi; });
    return yd;
  }

  std::vector<float> _decision_function_float(std::vector<float> x) const{
    /* Do the prediction with data in/out as float, cast to T, U before prediction */
    std::vector<T> xt;
    std::transform(x.begin(), x.end(), std::back_inserter(xt),
                   [](float xi) -> T { return (T) xi; });
    std::vector<U> y = decision_function(xt);
    std::vector<float> yd;
    std::transform(y.begin(), y.end(), std::back_inserter(yd),
                [](U yi) -> float { return (float) yi; });
    return yd;
  }

}; // class BDT

} // namespace conifer

#endif
