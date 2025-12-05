// sherpa-onnx/csrc/axcl/offline-sense-voice-model-axcl.cc
//
// Copyright (c)  2025  M5Stack Technology CO LTD

#include "sherpa-onnx/csrc/axcl/offline-sense-voice-model-axcl.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/axcl/acl-model.h"
#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {

class OfflineSenseVoiceModelAxcl::Impl {
 public:
  explicit Impl(const OfflineModelConfig &config) : config_(config) {
    model_ = std::make_unique<AxclModel>(config_.sense_voice.model_);

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
    std::array<int32_t, 4> prompt{language, 1, 2, text_norm};

    model_->SetInputTensorData("x", features.data(), features.size());
    model_->SetInputTensorData("prompt", prompt.data(), prompt.size());
    model_->Run();
    return GetOutputTensorData("logits");
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

  std::vector<float> ApplyLFR(std::vector<float> in) const {
    int32_t lfr_window_size = meta_data_.window_size;
    int32_t lfr_window_shift = meta_data_.window_shift;
    int32_t in_feat_dim = 80;
    int32_t in_num_frames = in.size() / in_feat_dim;
    int32_t out_num_frames =
        (in_num_frames - lfr_window_size) / lfr_window_shift + 1;

    if (out_num_frames > num_input_frames_) {
      SHERPA_ONNX_LOGE(
          "Number of input frames %d is too large. Truncate it to %d frames.",
          out_num_frames, num_input_frames_);
      SHERPA_ONNX_LOGE(
          "Recognition result may be truncated/incomplete. Please select a "
          "model accepting longer audios.");
      out_num_frames = num_input_frames_;
    }

    int32_t out_feat_dim = in_feat_dim * lfr_window_size;
    std::vector<float> out(num_input_frames_ * out_feat_dim);
    const float *p_in = in.data();
    float *p_out = out.data();
    for (int32_t i = 0; i != out_num_frames; ++i) {
      std::copy(p_in, p_in + out_feat_dim, p_out);
      p_out += out_feat_dim;
      p_in += lfr_window_shift * in_feat_dim;
    }
    return out;
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
