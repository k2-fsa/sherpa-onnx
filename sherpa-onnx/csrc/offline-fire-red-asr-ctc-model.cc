// sherpa-onnx/csrc/offline-fire-red-asr-ctc-model.cc
//
// Copyright (c)  2026  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-fire-red-asr-ctc-model.h"

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#if __OHOS__
#include "rawfile/raw_file_manager.h"
#endif

#include "Eigen/Dense"
#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/onnx-utils.h"
#include "sherpa-onnx/csrc/session.h"
#include "sherpa-onnx/csrc/text-utils.h"

namespace sherpa_onnx {

class OfflineFireRedAsrCtcModel::Impl {
 public:
  explicit Impl(const OfflineModelConfig &config)
      : config_(config),
        env_(ORT_LOGGING_LEVEL_ERROR),
        sess_opts_(GetSessionOptions(config)),
        allocator_{} {
    sess_ = std::make_unique<Ort::Session>(
        env_, SHERPA_ONNX_TO_ORT_PATH(config.fire_red_asr_ctc.model),
        sess_opts_);
    Init(nullptr, 0);
  }

  template <typename Manager>
  Impl(Manager *mgr, const OfflineModelConfig &config)
      : config_(config),
        env_(ORT_LOGGING_LEVEL_ERROR),
        sess_opts_(GetSessionOptions(config)),
        allocator_{} {
    auto buf = ReadFile(mgr, config_.fire_red_asr_ctc.model);
    Init(buf.data(), buf.size());
  }

  std::vector<Ort::Value> Forward(Ort::Value features,
                                  Ort::Value features_length) {
    std::array<Ort::Value, 2> inputs = {std::move(features),
                                        std::move(features_length)};

    return sess_->Run({}, input_names_ptr_.data(), inputs.data(), inputs.size(),
                      output_names_ptr_.data(), output_names_ptr_.size());
  }

  int32_t VocabSize() const { return vocab_size_; }

  int32_t SubsamplingFactor() const { return subsampling_factor_; }

  OrtAllocator *Allocator() { return allocator_; }

  void NormalizeFeatures(float *features, int32_t num_frames,
                         int32_t feat_dim) const {
    if (static_cast<int32_t>(mean_.size()) != feat_dim) {
      SHERPA_ONNX_LOGE("Bad things happened");
      SHERPA_ONNX_LOGE("Wrong feat dim %d. Expect: %d", feat_dim,
                       static_cast<int32_t>(mean_.size()));
      SHERPA_ONNX_EXIT(-1);
    }

    using RowMajorMat =
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    Eigen::Map<RowMajorMat> x(features, num_frames, feat_dim);

    Eigen::Map<const Eigen::RowVectorXf> mean(mean_.data(), feat_dim);
    Eigen::Map<const Eigen::RowVectorXf> inv_std(inv_stddev_.data(), feat_dim);
    x.array() =
        (x.array().rowwise() - mean.array()).rowwise() * inv_std.array();
  }

 private:
  void Init(void *model_data, size_t model_data_length) {
    if (model_data) {
      sess_ = std::make_unique<Ort::Session>(env_, model_data,
                                             model_data_length, sess_opts_);
    } else if (!sess_) {
      SHERPA_ONNX_LOGE(
          "Please pass buffer data or initialize session outside of this "
          "function");
      SHERPA_ONNX_EXIT(-1);
    }

    GetInputNames(sess_.get(), &input_names_, &input_names_ptr_);

    GetOutputNames(sess_.get(), &output_names_, &output_names_ptr_);

    // get meta data
    Ort::ModelMetadata meta_data = sess_->GetModelMetadata();
    if (config_.debug) {
      std::ostringstream os;
      PrintModelMetadata(os, meta_data);
#if __OHOS__
      SHERPA_ONNX_LOGE("%{public}s\n", os.str().c_str());
#else
      SHERPA_ONNX_LOGE("%s\n", os.str().c_str());
#endif
    }

    Ort::AllocatorWithDefaultOptions allocator;  // used in the macro below

    std::string model_type;
    SHERPA_ONNX_READ_META_DATA_STR(model_type, "model_type");
    if (model_type != "fire-red-asr-2-ctc") {
      SHERPA_ONNX_LOGE("Expect model type fire-red-asr-2-ctc. Given: '%s'",
                       model_type.c_str());
      SHERPA_ONNX_EXIT(-1);
    }

    SHERPA_ONNX_READ_META_DATA_WITH_DEFAULT(subsampling_factor_,
                                            "subsampling_factor", 4);

    auto shape =
        sess_->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    vocab_size_ = shape.back();

    if (config_.debug) {
#if __OHOS__
      SHERPA_ONNX_LOGE("subsampling_factor: %{public}d", subsampling_factor_);
      SHERPA_ONNX_LOGE("vocab_size: %{public}d", vocab_size_);
#else
      SHERPA_ONNX_LOGE("subsampling_factor: %d", subsampling_factor_);
      SHERPA_ONNX_LOGE("vocab_size: %d", vocab_size_);
#endif
    }

    SHERPA_ONNX_READ_META_DATA_VEC_FLOAT(mean_, "cmvn_mean");
    SHERPA_ONNX_READ_META_DATA_VEC_FLOAT(inv_stddev_, "cmvn_inv_stddev");
    if (mean_.size() != inv_stddev_.size()) {
      SHERPA_ONNX_LOGE("Incorrect cmvn. mean size: %d, inv_stddev size: %d",
                       static_cast<int32_t>(mean_.size()),
                       static_cast<int32_t>(inv_stddev_.size()));
      SHERPA_ONNX_EXIT(-1);
    }
  }

 private:
  OfflineModelConfig config_;
  Ort::Env env_;
  Ort::SessionOptions sess_opts_;
  Ort::AllocatorWithDefaultOptions allocator_;

  std::unique_ptr<Ort::Session> sess_;

  std::vector<std::string> input_names_;
  std::vector<const char *> input_names_ptr_;

  std::vector<std::string> output_names_;
  std::vector<const char *> output_names_ptr_;

  int32_t vocab_size_ = 0;
  int32_t subsampling_factor_ = 0;

  std::vector<float> mean_;
  std::vector<float> inv_stddev_;
};

OfflineFireRedAsrCtcModel::OfflineFireRedAsrCtcModel(
    const OfflineModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

template <typename Manager>
OfflineFireRedAsrCtcModel::OfflineFireRedAsrCtcModel(
    Manager *mgr, const OfflineModelConfig &config)
    : impl_(std::make_unique<Impl>(mgr, config)) {}

OfflineFireRedAsrCtcModel::~OfflineFireRedAsrCtcModel() = default;

std::vector<Ort::Value> OfflineFireRedAsrCtcModel::Forward(
    Ort::Value features, Ort::Value features_length) {
  return impl_->Forward(std::move(features), std::move(features_length));
}

int32_t OfflineFireRedAsrCtcModel::VocabSize() const {
  return impl_->VocabSize();
}

int32_t OfflineFireRedAsrCtcModel::SubsamplingFactor() const {
  return impl_->SubsamplingFactor();
}

OrtAllocator *OfflineFireRedAsrCtcModel::Allocator() const {
  return impl_->Allocator();
}

void OfflineFireRedAsrCtcModel::NormalizeFeatures(float *features,
                                                  int32_t num_frames,
                                                  int32_t feat_dim) const {
  return impl_->NormalizeFeatures(features, num_frames, feat_dim);
}

#if __ANDROID_API__ >= 9
template OfflineFireRedAsrCtcModel::OfflineFireRedAsrCtcModel(
    AAssetManager *mgr, const OfflineModelConfig &config);
#endif

#if __OHOS__
template OfflineFireRedAsrCtcModel::OfflineFireRedAsrCtcModel(
    NativeResourceManager *mgr, const OfflineModelConfig &config);
#endif

}  // namespace sherpa_onnx
