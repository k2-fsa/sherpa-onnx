// sherpa-onnx/csrc/offline-dolphin-model.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-dolphin-model.h"

#include <algorithm>
#include <string>
#include <utility>

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#if __OHOS__
#include "rawfile/raw_file_manager.h"
#endif

#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/onnx-utils.h"
#include "sherpa-onnx/csrc/session.h"
#include "sherpa-onnx/csrc/text-utils.h"

namespace sherpa_onnx {

class OfflineDolphinModel::Impl {
 public:
  explicit Impl(const OfflineModelConfig &config)
      : config_(config),
        env_(ORT_LOGGING_LEVEL_ERROR),
        sess_opts_(GetSessionOptions(config)),
        allocator_{} {
    auto buf = ReadFile(config_.dolphin.model);
    Init(buf.data(), buf.size());
  }

  template <typename Manager>
  Impl(Manager *mgr, const OfflineModelConfig &config)
      : config_(config),
        env_(ORT_LOGGING_LEVEL_ERROR),
        sess_opts_(GetSessionOptions(config)),
        allocator_{} {
    auto buf = ReadFile(mgr, config_.dolphin.model);
    Init(buf.data(), buf.size());
  }

  std::vector<Ort::Value> Forward(Ort::Value features,
                                  Ort::Value features_length) {
    std::array<Ort::Value, 2> inputs = {
        std::move(features),
        std::move(features_length),
    };

    return sess_->Run({}, input_names_ptr_.data(), inputs.data(), inputs.size(),
                      output_names_ptr_.data(), output_names_ptr_.size());
  }

  int32_t VocabSize() const { return meta_data_.vocab_size; }

  int32_t SubsamplingFactor() const { return meta_data_.subsampling_factor; }

  void NormalizeFeatures(float *features, int32_t num_frames,
                         int32_t feat_dim) const {
    auto p = features;
    const auto &mean = meta_data_.mean;
    const auto &invstd = meta_data_.inv_stddev;

    for (int32_t f = 0; f < num_frames; ++f) {
      for (int32_t d = 0; d < feat_dim; ++d) {
        p[d] = (p[d] - mean[d]) * invstd[d];
      }
      p += feat_dim;
    }
  }

  OrtAllocator *Allocator() { return allocator_; }

 private:
  void Init(void *model_data, size_t model_data_length) {
    sess_ = std::make_unique<Ort::Session>(env_, model_data, model_data_length,
                                           sess_opts_);

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
    SHERPA_ONNX_READ_META_DATA(meta_data_.vocab_size, "vocab_size");

    SHERPA_ONNX_READ_META_DATA_VEC_FLOAT(meta_data_.mean, "mean");
    SHERPA_ONNX_READ_META_DATA_VEC_FLOAT(meta_data_.inv_stddev, "invstd");
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

  OfflineDolphinModelMetaData meta_data_;
};

OfflineDolphinModel::OfflineDolphinModel(const OfflineModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

template <typename Manager>
OfflineDolphinModel::OfflineDolphinModel(Manager *mgr,
                                         const OfflineModelConfig &config)
    : impl_(std::make_unique<Impl>(mgr, config)) {}

OfflineDolphinModel::~OfflineDolphinModel() = default;

std::vector<Ort::Value> OfflineDolphinModel::Forward(
    Ort::Value features, Ort::Value features_length) {
  return impl_->Forward(std::move(features), std::move(features_length));
}

int32_t OfflineDolphinModel::VocabSize() const { return impl_->VocabSize(); }

int32_t OfflineDolphinModel::SubsamplingFactor() const {
  return impl_->SubsamplingFactor();
}

void OfflineDolphinModel::NormalizeFeatures(float *features, int32_t num_frames,
                                            int32_t feat_dim) const {
  return impl_->NormalizeFeatures(features, num_frames, feat_dim);
}

OrtAllocator *OfflineDolphinModel::Allocator() const {
  return impl_->Allocator();
}

#if __ANDROID_API__ >= 9
template OfflineDolphinModel::OfflineDolphinModel(
    AAssetManager *mgr, const OfflineModelConfig &config);
#endif

#if __OHOS__
template OfflineDolphinModel::OfflineDolphinModel(
    NativeResourceManager *mgr, const OfflineModelConfig &config);
#endif

}  // namespace sherpa_onnx
