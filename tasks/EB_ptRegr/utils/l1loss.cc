
#include <dmlc/omp.h>

#include <algorithm>
#include <cmath>
#include <cstdint>  // std::int32_t
#include <vector>
#include <iostream>

#include "../../src/common/common.h"
#include "../../src/common/linalg_op.h"
#include "../../src/common/numeric.h"          // Reduce
#include "../../src/common/optional_weight.h"  // OptionalWeights
#include "../../src/common/pseudo_huber.h"
#include "../../src/common/stats.h"
#include "../../src/common/threading_utils.h"
#include "../../src/common/transform.h"
#include "../../src/objective/regression_loss.h"
#include "../../src/objective/adaptive.h"
#include "../../src/objective/init_estimation.h"  // FitIntercept
#include "xgboost/base.h"
#include "xgboost/context.h"  // Context
#include "xgboost/data.h"     // MetaInfo
#include "xgboost/host_device_vector.h"
#include "xgboost/json.h"
#include "xgboost/linalg.h"
#include "xgboost/logging.h"
#include "xgboost/objective.h"  // ObjFunction
#include "xgboost/parameter.h"
#include "xgboost/span.h"
#include "xgboost/tree_model.h"  // RegTree

#include "../../src/objective/regression_param.h"

#if defined(XGBOOST_USE_CUDA)
#include "../../src/common/cuda_context.cuh"  // for CUDAContext
#include "../../src/common/device_helpers.cuh"
#include "../../src/common/linalg_op.cuh"
#endif  // defined(XGBOOST_USE_CUDA)

#if defined(XGBOOST_USE_SYCL)
#include "../../plugin/sycl/common/linalg_op.h"
#endif




namespace xgboost::obj {
// This is a helpful data structure to define parameters
// You do not have to use it.
// see http://dmlc-core.readthedocs.org/en/latest/parameter.html
// for introduction of this module.
struct L1LossParam : public XGBoostParameter<L1LossParam> {
  float alphaL1;
  float betaL1;
  float pt_thr;
  float bkg_target;
  std::string cls_s;
  std::vector<float> cls;
  std::string pts_s;
  std::vector<float> pts;

  DMLC_DECLARE_PARAMETER(L1LossParam) {
    DMLC_DECLARE_FIELD(alphaL1).set_default(1.0f).set_lower_bound(0.0f)
        .describe("Penalty term for pred>1 for the background class.");
    DMLC_DECLARE_FIELD(betaL1).set_default(0.0f).set_lower_bound(0.0f)
        .describe("Scaling of the pt penalty term for the background");
    DMLC_DECLARE_FIELD(pt_thr).set_default(100.0f).set_lower_bound(0.0f)
        .describe("Threshold of the pt term");
    DMLC_DECLARE_FIELD(bkg_target).set_default(1.0f).set_lower_bound(0.0f)
        .describe("Output target for the background class. ");
    DMLC_DECLARE_FIELD(cls_s)
        .set_default({}) // empty by default
        .describe("Class label vector comma separated string");
    DMLC_DECLARE_FIELD(pts_s)
        .set_default({}) // empty by default
        .describe("pt vector comma separated string");
  }
};

DMLC_REGISTER_PARAMETER(L1LossParam);

namespace {
void CheckRegInputs(MetaInfo const& info, HostDeviceVector<bst_float> const& preds) {
  CheckInitInputs(info);
  CHECK_EQ(info.labels.Size(), preds.Size()) << "Invalid shape of labels.";
}

template <typename Loss>
void ValidateLabel(Context const* ctx, MetaInfo const& info) {
  auto label = info.labels.View(ctx->Device());
  auto valid = ctx->DispatchDevice(
      [&] {
        return std::all_of(linalg::cbegin(label), linalg::cend(label),
                           [](float y) -> bool { return Loss::CheckLabel(y); });
      },
      [&] {
#if defined(XGBOOST_USE_CUDA)
        auto cuctx = ctx->CUDACtx();
        auto it = dh::MakeTransformIterator<bool>(
            thrust::make_counting_iterator(0ul), [=] XGBOOST_DEVICE(std::size_t i) -> bool {
              auto [m, n] = linalg::UnravelIndex(i, label.Shape());
              return Loss::CheckLabel(label(m, n));
            });
        return dh::Reduce(cuctx->CTP(), it, it + label.Size(), true, thrust::logical_and<>{});
#else
        common::AssertGPUSupport();
        return false;
#endif  // defined(XGBOOST_USE_CUDA)
      },
      [&] {
#if defined(XGBOOST_USE_SYCL)
        return sycl::linalg::Validate(ctx->Device(), label,
                                      [](float y) -> bool { return Loss::CheckLabel(y); });
#else
        common::AssertSYCLSupport();
        return false;
#endif  // defined(XGBOOST_USE_SYCL)
      });
  if (!valid) {
    LOG(FATAL) << Loss::LabelErrorMsg();
  }
}
}  // anonymous namespace

class L1Loss : public ObjFunction {
 public:
  void Configure(const Args& args) override {
    param_.UpdateAllowUnknown(args);
    //split the cls_s string into a vector of strings
    std::vector<std::string> cls_s_split = common::Split(param_.cls_s, ',');
    std::vector<std::string> pts_s_split = common::Split(param_.pts_s, ',');

    param_.cls.resize(cls_s_split.size());
    param_.pts.resize(pts_s_split.size());
    for(int i=0; i< cls_s_split.size(); i++){
      param_.cls[i]= (float)std::stod(cls_s_split[i]);
      param_.pts[i]= (float)std::stod(pts_s_split[i]);
    }
  }
  [[nodiscard]] ObjInfo Task() const override { return {ObjInfo::kRegression, true, true}; }
  [[nodiscard]] bst_target_t Targets(MetaInfo const& info) const override {
    return static_cast<std::size_t>(1);
  }

  void GetGradient(HostDeviceVector<float> const& preds, const MetaInfo& info,
                   std::int32_t /*iter*/, linalg::Matrix<GradientPair>* out_gpair) override {
    CheckRegInputs(info, preds);
    auto labels = info.labels.View(ctx_->Device());

    out_gpair->SetDevice(ctx_->Device());
    out_gpair->Reshape(info.num_row_, this->Targets(info));
    auto gpair = out_gpair->View(ctx_->Device());

    preds.SetDevice(ctx_->Device());
    auto predt = linalg::MakeTensorView(ctx_, &preds, info.num_row_, this->Targets(info));
    info.weights_.SetDevice(ctx_->Device());
    common::OptionalWeights weight{ctx_->IsCPU() ? info.weights_.ConstHostSpan()
                                                 : info.weights_.ConstDeviceSpan()};

    linalg::ElementWiseKernel(
        ctx_, labels, [=] XGBOOST_DEVICE(std::size_t i, std::size_t j) mutable {
          auto sign = [](auto x) {
            return (x > static_cast<decltype(x)>(0)) - (x < static_cast<decltype(x)>(0));
          };
          auto y = labels(i, j);
          auto cls = param_.cls[i];
          auto pt = param_.pts[i];

          auto hess = weight[i] * (cls + (1-cls) *  (param_.alphaL1 + param_.betaL1 * std::min(pt,param_.pt_thr) ));
          auto grad = weight[i]  * (cls  * sign(predt(i, j) - y) + (1-cls) * (param_.alphaL1 + param_.betaL1 * std::min(pt, param_.pt_thr)) * (predt(i, j) > param_.bkg_target));
          gpair(i, j) = GradientPair{grad, hess};
        });
  }

  void InitEstimation(MetaInfo const& info, linalg::Tensor<float, 1>* base_margin) const override {
    CheckInitInputs(info);
    base_margin->Reshape(this->Targets(info));

    double w{0.0};
    if (info.weights_.Empty()) {
      w = static_cast<double>(info.num_row_);
    } else {
      w = common::Reduce(ctx_, info.weights_);
    }

    if (info.num_row_ == 0) {
      auto out = base_margin->HostView();
      out(0) = 0;
    } else {
      linalg::Vector<float> temp;
      common::Median(ctx_, info.labels, info.weights_, &temp);
      common::Mean(ctx_, temp, base_margin);
    }
    CHECK_EQ(base_margin->Size(), 1);
    auto out = base_margin->HostView();
    // weighted avg
    std::transform(linalg::cbegin(out), linalg::cend(out), linalg::begin(out),
                   [w](float v) { return v * w; });

    auto rc = collective::Success() << [&] {
      return collective::GlobalSum(ctx_, info, out);
    } << [&] {
      return collective::GlobalSum(ctx_, info, linalg::MakeVec(&w, 1));
    };
    collective::SafeColl(rc);

    if (common::CloseTo(w, 0.0)) {
      // Mostly for handling empty dataset test.
      LOG(WARNING) << "Sum of weights is close to 0.0, skipping base score estimation.";
      out(0) = ObjFunction::DefaultBaseScore();
      return;
    }
    std::transform(linalg::cbegin(out), linalg::cend(out), linalg::begin(out),
                   [w](float v) { return v / w; });
  }

  void UpdateTreeLeaf(HostDeviceVector<bst_node_t> const& position, MetaInfo const& info,
                      float learning_rate, HostDeviceVector<float> const& prediction,
                      std::int32_t group_idx, RegTree* p_tree) const override {
    ::xgboost::obj::UpdateTreeLeaf(ctx_, position, group_idx, info, learning_rate, prediction, 0.5,
                                   p_tree);
  }

  [[nodiscard]] const char* DefaultEvalMetric() const override { return "mae"; }

  void SaveConfig(Json* p_out) const override {
    auto& out = *p_out;
    out["my_param"] = ToJson(param_);
    out["name"] = String("reg:l1loss");
  }

  void LoadConfig(Json const& in) override {
    FromJson(in["my_param"], &param_);
    CHECK_EQ(StringView{get<String const>(in["name"])}, StringView{"reg:l1loss"});
  }
  private:
    L1LossParam param_;
};

XGBOOST_REGISTER_OBJECTIVE(L1Loss, "reg:l1loss")
    .describe("Mean absoluate error with penalty on the bkg class.")
    .set_body([]() { return new L1Loss(); });
}  // namespace xgboost::obj