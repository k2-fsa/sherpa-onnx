// sherpa-onnx/csrc/online-speech-denoiser-gtcrn-impl.h
//
// Copyright (c)  2026  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_ONLINE_SPEECH_DENOISER_GTCRN_IMPL_H_
#define SHERPA_ONNX_CSRC_ONLINE_SPEECH_DENOISER_GTCRN_IMPL_H_

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <utility>

#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/offline-speech-denoiser-gtcrn-model.h"
#include "sherpa-onnx/csrc/online-speech-denoiser-impl.h"
#include "sherpa-onnx/csrc/online-speech-denoiser-stft-impl.h"

namespace sherpa_onnx {

class OnlineSpeechDenoiserGtcrnImpl : public OnlineSpeechDenoiserImpl {
 public:
  explicit OnlineSpeechDenoiserGtcrnImpl(
      const OnlineSpeechDenoiserConfig &config)
      : model_(config.model),
        stream_(GetStftConfig(model_.GetMetaData())),
        states_(model_.GetInitStates()) {}

  template <typename Manager>
  OnlineSpeechDenoiserGtcrnImpl(Manager *mgr,
                                const OnlineSpeechDenoiserConfig &config)
      : model_(mgr, config.model),
        stream_(GetStftConfig(model_.GetMetaData())),
        states_(model_.GetInitStates()) {}

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
        [this]() { states_ = model_.GetInitStates(); });
  }

  void Reset() override {
    stream_.Reset();
    states_ = model_.GetInitStates();
  }

  int32_t GetSampleRate() const override { return stream_.GetSampleRate(); }

  int32_t GetFrameShiftInSamples() const override {
    return stream_.GetFrameShiftInSamples();
  }

 private:
  static OnlineSpeechDenoiserStftConfig GetStftConfig(
      const OfflineSpeechDenoiserGtcrnModelMetaData &meta) {
    OnlineSpeechDenoiserStftConfig config;
    config.sample_rate = meta.sample_rate;
    config.n_fft = meta.n_fft;
    config.hop_length = meta.hop_length;
    config.window_length = meta.window_length;
    config.window_type = meta.window_type;
    return config;
  }

  void ProcessFrame(float *spec, size_t spec_size, float *enhanced) {
    const int32_t num_bins = stream_.GetNumBins();
    const size_t expected_size = static_cast<size_t>(num_bins * 2);
    if (spec_size != expected_size) {
      SHERPA_ONNX_LOGE("Unexpected GTCRN spec size. Expected: %d. Given: %d",
                       num_bins * 2, static_cast<int32_t>(spec_size));
      SHERPA_ONNX_EXIT(-1);
    }

    std::array<int64_t, 4> x_shape{1, num_bins, 1, 2};
    Ort::Value x_tensor = Ort::Value::CreateTensor<float>(
        stream_.GetMemoryInfo(), spec, spec_size, x_shape.data(),
        x_shape.size());

    Ort::Value output{nullptr};
    std::tie(output, states_) =
        model_.Run(std::move(x_tensor), std::move(states_));

    const float *enhanced_spec = output.GetTensorData<float>();
    std::copy(enhanced_spec, enhanced_spec + spec_size, enhanced);
  }

 private:
  OfflineSpeechDenoiserGtcrnModel model_;
  OnlineSpeechDenoiserStftImpl stream_;
  OfflineSpeechDenoiserGtcrnModel::States states_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_ONLINE_SPEECH_DENOISER_GTCRN_IMPL_H_
