// sherpa-onnx/csrc/online-speech-denoiser-dpdfnet-impl.h
//
// Copyright (c)  2026  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_ONLINE_SPEECH_DENOISER_DPDFNET_IMPL_H_
#define SHERPA_ONNX_CSRC_ONLINE_SPEECH_DENOISER_DPDFNET_IMPL_H_

#include <algorithm>
#include <cstdint>
#include <memory>
#include <utility>

#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/offline-speech-denoiser-dpdfnet-model.h"
#include "sherpa-onnx/csrc/online-speech-denoiser-impl.h"
#include "sherpa-onnx/csrc/online-speech-denoiser-stft-impl.h"

namespace sherpa_onnx {

class OnlineSpeechDenoiserDpdfNetImpl : public OnlineSpeechDenoiserImpl {
 public:
  explicit OnlineSpeechDenoiserDpdfNetImpl(
      const OnlineSpeechDenoiserConfig &config)
      : model_(config.model),
        stream_(GetStftConfig(model_.GetMetaData())),
        state_(model_.GetInitState()) {
    Init();
  }

  template <typename Manager>
  OnlineSpeechDenoiserDpdfNetImpl(Manager *mgr,
                                  const OnlineSpeechDenoiserConfig &config)
      : model_(mgr, config.model),
        stream_(GetStftConfig(model_.GetMetaData())),
        state_(model_.GetInitState()) {
    Init();
  }

  DenoisedAudio Run(const float *samples, int32_t n,
                    int32_t sample_rate) override {
    return stream_.Run(samples, n, sample_rate,
                       [this](float *spec, size_t spec_size, float *enhanced) {
                         ProcessFrame(spec, spec_size, enhanced);
                       });
  }

  DenoisedAudio Flush() override {
    return stream_.Flush(
        [this](float *spec, size_t spec_size, float *enhanced) {
          ProcessFrame(spec, spec_size, enhanced);
        },
        [this]() { state_ = model_.GetInitState(); });
  }

  void Reset() override {
    stream_.Reset();
    state_ = model_.GetInitState();
  }

  int32_t GetSampleRate() const override { return stream_.GetSampleRate(); }

  int32_t GetFrameShiftInSamples() const override {
    return stream_.GetFrameShiftInSamples();
  }

 private:
  void Init() {
    const auto &meta = model_.GetMetaData();
    if (meta.profile != "dpdfnet_16khz" &&
        meta.profile != "dpdfnet2_48khz_hr") {
      SHERPA_ONNX_LOGE(
          "Online speech denoiser currently supports only DPDFNet streaming "
          "exports. Given profile: %s",
          meta.profile.c_str());
      SHERPA_ONNX_EXIT(-1);
    }

    if (meta.spec_shape.size() != 4 || meta.spec_shape[0] != 1 ||
        meta.spec_shape[1] != 1 || meta.spec_shape[3] != 2) {
      SHERPA_ONNX_LOGE(
          "Online speech denoiser expects a single-frame DPDFNet ONNX "
          "signature shaped like [1, 1, F, 2].");
      SHERPA_ONNX_EXIT(-1);
    }
  }

  static OnlineSpeechDenoiserStftConfig GetStftConfig(
      const OfflineSpeechDenoiserDpdfNetModelMetaData &meta) {
    OnlineSpeechDenoiserStftConfig config;
    config.sample_rate = meta.sample_rate;
    config.n_fft = meta.n_fft;
    config.hop_length = meta.hop_length;
    config.window_length = meta.window_length;
    config.window_type = meta.window_type;
    return config;
  }

  void ProcessFrame(float *spec, size_t spec_size, float *enhanced) {
    const auto &meta = model_.GetMetaData();
    const int32_t expected_size = meta.spec_shape[2] * meta.spec_shape[3];
    if (spec_size != static_cast<size_t>(expected_size)) {
      SHERPA_ONNX_LOGE("Unexpected DPDFNet spec size. Expected: %d. Given: %d",
                       expected_size, static_cast<int32_t>(spec_size));
      SHERPA_ONNX_EXIT(-1);
    }

    Ort::Value spec_tensor = Ort::Value::CreateTensor<float>(
        stream_.GetMemoryInfo(), spec, spec_size, meta.spec_shape.data(),
        meta.spec_shape.size());

    auto out = model_.Run(std::move(spec_tensor), std::move(state_));
    state_ = std::move(out.second);

    const float *enhanced_spec = out.first.GetTensorData<float>();
    std::copy(enhanced_spec, enhanced_spec + spec_size, enhanced);
  }

 private:
  OfflineSpeechDenoiserDpdfNetModel model_;
  OnlineSpeechDenoiserStftImpl stream_;
  Ort::Value state_{nullptr};
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_ONLINE_SPEECH_DENOISER_DPDFNET_IMPL_H_
