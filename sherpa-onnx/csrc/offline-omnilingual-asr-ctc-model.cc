// sherpa-onnx/csrc/offline-omnilingual-asr-ctc-model.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-omnilingual-asr-ctc-model.h"

#include <algorithm>
#include <cmath>
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

#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/onnx-utils.h"
#include "sherpa-onnx/csrc/session.h"
#include "sherpa-onnx/csrc/text-utils.h"
#include "sherpa-onnx/csrc/transpose.h"

namespace sherpa_onnx {

class OfflineOmnilingualAsrCtcModel::Impl {
 public:
  explicit Impl(const OfflineModelConfig &config)
      : config_(config),
        env_(ORT_LOGGING_LEVEL_ERROR),
        sess_opts_(GetSessionOptions(config)),
        allocator_{} {
    sess_ = std::make_unique<Ort::Session>(
        env_, SHERPA_ONNX_TO_ORT_PATH(config_.omnilingual.model), sess_opts_);
    Init(nullptr, 0);
  }

  template <typename Manager>
  Impl(Manager *mgr, const OfflineModelConfig &config)
      : config_(config),
        env_(ORT_LOGGING_LEVEL_ERROR),
        sess_opts_(GetSessionOptions(config)),
        allocator_{} {
    auto buf = ReadFile(mgr, config_.omnilingual.model);
    Init(buf.data(), buf.size());
  }

  std::vector<Ort::Value> Forward(Ort::Value features,
                                  Ort::Value /*/features_length*/) {
    auto out_vec =
        sess_->Run({}, input_names_ptr_.data(), &features, 1,
                   output_names_ptr_.data(), output_names_ptr_.size());
    std::vector<int64_t> logits_shape =
        out_vec[0].GetTensorTypeAndShapeInfo().GetShape();

    std::vector<int64_t> num_frames(logits_shape[0], logits_shape[1]);

    int64_t shape = logits_shape[0];

    Ort::Value logits_len =
        Ort::Value::CreateTensor<int64_t>(allocator_, &shape, 1);
    std::copy(num_frames.begin(), num_frames.end(),
              logits_len.GetTensorMutableData<int64_t>());

    out_vec.push_back(std::move(logits_len));

    return out_vec;
  }

  int32_t VocabSize() const { return vocab_size_; }

  OrtAllocator *Allocator() { return allocator_; }

  static void NormalizeFeatures(float *features, int32_t num_frames,
                                int32_t feat_dim) {
    if (num_frames != 1) {
      SHERPA_ONNX_LOGE(
          "Unexpected error in collecting samples for Omnilingual ASR models!");
      return;
    }

    double s = 0;
    double sq = 0;
    for (int32_t i = 0; i < feat_dim; ++i) {
      s += features[i];
      sq += features[i] * features[i];
    }

    double mean = s / feat_dim;
    double inv_stddev = 1 / std::sqrt(sq / feat_dim - mean * mean + 1e-5);

    for (int32_t i = 0; i < feat_dim; ++i) {
      features[i] = (features[i] - mean) * inv_stddev;
    }
  }

 private:
  void Init(void *model_data, size_t model_data_length) {
    // For models with 1B parameters, weights are saved externally
    // in model.weights
    // We cannot create session from buffer in this case.
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

    // get vocab size from the output[0].shape, which is (N, T, vocab_size)
    vocab_size_ =
        sess_->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape()[2];
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
};

OfflineOmnilingualAsrCtcModel::OfflineOmnilingualAsrCtcModel(
    const OfflineModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

template <typename Manager>
OfflineOmnilingualAsrCtcModel::OfflineOmnilingualAsrCtcModel(
    Manager *mgr, const OfflineModelConfig &config)
    : impl_(std::make_unique<Impl>(mgr, config)) {}

OfflineOmnilingualAsrCtcModel::~OfflineOmnilingualAsrCtcModel() = default;

std::vector<Ort::Value> OfflineOmnilingualAsrCtcModel::Forward(
    Ort::Value features, Ort::Value features_length) {
  return impl_->Forward(std::move(features), std::move(features_length));
}

int32_t OfflineOmnilingualAsrCtcModel::VocabSize() const {
  return impl_->VocabSize();
}

OrtAllocator *OfflineOmnilingualAsrCtcModel::Allocator() const {
  return impl_->Allocator();
}

void OfflineOmnilingualAsrCtcModel::NormalizeFeatures(float *features,
                                                      int32_t num_frames,
                                                      int32_t feat_dim) const {
  return impl_->NormalizeFeatures(features, num_frames, feat_dim);
}

#if __ANDROID_API__ >= 9
template OfflineOmnilingualAsrCtcModel::OfflineOmnilingualAsrCtcModel(
    AAssetManager *mgr, const OfflineModelConfig &config);
#endif

#if __OHOS__
template OfflineOmnilingualAsrCtcModel::OfflineOmnilingualAsrCtcModel(
    NativeResourceManager *mgr, const OfflineModelConfig &config);
#endif

}  // namespace sherpa_onnx
