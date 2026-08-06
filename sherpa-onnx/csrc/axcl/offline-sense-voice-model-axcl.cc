// sherpa-onnx/csrc/axcl/offline-sense-voice-model-axcl.cc
//
// Copyright (c)  2025  M5Stack Technology CO LTD

#include "sherpa-onnx/csrc/axcl/offline-sense-voice-model-axcl.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/axcl/axcl-model.h"
#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/lfr.h"
#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {

class OfflineSenseVoiceModelAxcl::Impl {
 public:
  explicit Impl(const OfflineModelConfig &config) : config_(config) {
    model_ = std::make_unique<AxclModel>(config_.sense_voice.model);

    PostInit();
  }

  template <typename Manager>
  Impl(Manager *mgr, const OfflineModelConfig &config) : config_(config) {
    auto buf = ReadFile(mgr, config_.sense_voice.model);
    model_ = std::make_unique<AxclModel>(buf.data(), buf.size());

    PostInit();
  }

  const OfflineSenseVoiceModelMetaData &GetModelMetadata() const {
    return meta_data_;
  }

  std::vector<float> Run(std::vector<float> features, int32_t language,
                         int32_t text_norm) {
    features = ApplyLFR(std::move(features));
    if (features.empty()) {
      return {};
    }

    std::array<int32_t, 4> prompt{language, 1, 2, text_norm};

    model_->SetInputTensorData("x", features.data(), features.size());
    model_->SetInputTensorData("prompt", prompt.data(), prompt.size());
    model_->Run();
    return model_->GetOutputTensorData("logits");
  }

 private:
  void PostInit() {
    if (!model_->IsInitialized()) {
      SHERPA_ONNX_LOGE("Failed to initialize the model with '%s'",
                       config_.sense_voice.model.c_str());
      SHERPA_ONNX_EXIT(-1);
    }

    num_input_frames_ = model_->TensorShape("x")[1];

    if (config_.debug) {
      SHERPA_ONNX_LOGE("  num_input_frames_ = %d", num_input_frames_);
    }
  }

  std::vector<float> ApplyLFR(const std::vector<float> &in) const {
    return ApplyLfrForFixedShape(
        in, /*input_dim=*/80, meta_data_.window_size, meta_data_.window_shift,
        num_input_frames_);
  }

 private:
  OfflineModelConfig config_;
  std::unique_ptr<AxclModel> model_;
  OfflineSenseVoiceModelMetaData meta_data_;
  int32_t num_input_frames_ = -1;
};

OfflineSenseVoiceModelAxcl::~OfflineSenseVoiceModelAxcl() = default;

OfflineSenseVoiceModelAxcl::OfflineSenseVoiceModelAxcl(
    const OfflineModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

template <typename Manager>
OfflineSenseVoiceModelAxcl::OfflineSenseVoiceModelAxcl(
    Manager *mgr, const OfflineModelConfig &config)
    : impl_(std::make_unique<Impl>(mgr, config)) {}

std::vector<float> OfflineSenseVoiceModelAxcl::Run(std::vector<float> features,
                                                   int32_t language,
                                                   int32_t text_norm) const {
  return impl_->Run(std::move(features), language, text_norm);
}

const OfflineSenseVoiceModelMetaData &
OfflineSenseVoiceModelAxcl::GetModelMetadata() const {
  return impl_->GetModelMetadata();
}

#if __ANDROID_API__ >= 9
template OfflineSenseVoiceModelAxcl::OfflineSenseVoiceModelAxcl(
    AAssetManager *mgr, const OfflineModelConfig &config);
#endif

#if __OHOS__
template OfflineSenseVoiceModelAxcl::OfflineSenseVoiceModelAxcl(
    NativeResourceManager *mgr, const OfflineModelConfig &config);
#endif

}  // namespace sherpa_onnx
