// sherpa-onnx/csrc/offline-speech-denoiser-dpdfnet-impl.h
//
// Copyright (c)  2026  Ceva Inc

#ifndef SHERPA_ONNX_CSRC_OFFLINE_SPEECH_DENOISER_DPDFNET_IMPL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_SPEECH_DENOISER_DPDFNET_IMPL_H_

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <utility>
#include <vector>

#include "kaldi-native-fbank/csrc/istft.h"
#include "kaldi-native-fbank/csrc/stft.h"
#include "sherpa-onnx/csrc/math.h"
#include "sherpa-onnx/csrc/offline-speech-denoiser-dpdfnet-model.h"
#include "sherpa-onnx/csrc/offline-speech-denoiser-impl.h"
#include "sherpa-onnx/csrc/offline-speech-denoiser.h"
#include "sherpa-onnx/csrc/resample.h"

namespace sherpa_onnx {

class OfflineSpeechDenoiserDpdfNetImpl : public OfflineSpeechDenoiserImpl {
 public:
  explicit OfflineSpeechDenoiserDpdfNetImpl(
      const OfflineSpeechDenoiserConfig &config)
      : model_(config.model) {}

  template <typename Manager>
  OfflineSpeechDenoiserDpdfNetImpl(Manager *mgr,
                                   const OfflineSpeechDenoiserConfig &config)
      : model_(mgr, config.model) {}

  DenoisedAudio Run(const float *samples, int32_t n,
                    int32_t sample_rate) const override {
    const auto &meta = model_.GetMetaData();

    std::vector<float> tmp;
    auto p = samples;

    if (sample_rate != meta.sample_rate) {
      SHERPA_ONNX_LOGE(
          "Creating a resampler:\n"
          "   in_sample_rate: %d\n"
          "   output_sample_rate: %d\n",
          sample_rate, meta.sample_rate);

      float min_freq = std::min<int32_t>(sample_rate, meta.sample_rate);
      float lowpass_cutoff = 0.99f * 0.5f * min_freq;

      int32_t lowpass_filter_width = 6;
      auto resampler = std::make_unique<LinearResample>(
          sample_rate, meta.sample_rate, lowpass_cutoff, lowpass_filter_width);
      resampler->Resample(samples, n, true, &tmp);
      p = tmp.data();
      n = tmp.size();
    }

    auto stft_config = GetStftConfig();
    knf::Stft stft(stft_config);
    knf::StftResult stft_result = stft.Compute(p, n);

    auto state = model_.GetInitState();
    Ort::Value next_state{nullptr};

    knf::StftResult enhanced_stft_result;
    enhanced_stft_result.num_frames = stft_result.num_frames;
    for (int32_t i = 0; i < stft_result.num_frames; ++i) {
      auto frame = Process(stft_result, i, std::move(state), &next_state);
      state = std::move(next_state);

      enhanced_stft_result.real.insert(enhanced_stft_result.real.end(),
                                       frame.first.begin(), frame.first.end());
      enhanced_stft_result.imag.insert(enhanced_stft_result.imag.end(),
                                       frame.second.begin(),
                                       frame.second.end());
    }

    knf::IStft istft(stft_config);

    DenoisedAudio denoised_audio;
    denoised_audio.sample_rate = meta.sample_rate;
    denoised_audio.samples = ShiftWaveform(istft.Compute(enhanced_stft_result),
                                           meta.window_length * 2);
    return denoised_audio;
  }

  int32_t GetSampleRate() const override {
    return model_.GetMetaData().sample_rate;
  }

 private:
  static std::vector<float> ShiftWaveform(std::vector<float> samples,
                                          int32_t shift) {
    if (samples.size() > static_cast<size_t>(shift)) {
      std::copy(samples.begin() + shift, samples.end(), samples.begin());
      samples.resize(samples.size() - shift);
    } else {
      samples.clear();
    }

    samples.resize(samples.size() + shift, 0.0f);
    return samples;
  }

  knf::StftConfig GetStftConfig() const {
    const auto &meta = model_.GetMetaData();

    knf::StftConfig stft_config;
    stft_config.n_fft = meta.n_fft;
    stft_config.hop_length = meta.hop_length;
    stft_config.win_length = meta.window_length;
    stft_config.normalized = meta.normalized;
    stft_config.center = meta.center;
    stft_config.pad_mode = meta.pad_mode;
    stft_config.window_type = meta.window_type;
    stft_config.window = MakeVorbisWindow(meta.window_length);

    return stft_config;
  }

  std::pair<std::vector<float>, std::vector<float>> Process(
      const knf::StftResult &stft_result, int32_t frame_index, Ort::Value state,
      Ort::Value *next_state) const {
    const auto &meta = model_.GetMetaData();
    const int32_t n_fft = meta.n_fft;

    std::vector<float> x((n_fft / 2 + 1) * 2);

    const float *p_real =
        stft_result.real.data() + frame_index * (n_fft / 2 + 1);
    const float *p_imag =
        stft_result.imag.data() + frame_index * (n_fft / 2 + 1);

    for (int32_t i = 0; i < n_fft / 2 + 1; ++i) {
      x[2 * i] = p_real[i];
      x[2 * i + 1] = p_imag[i];
    }

    auto memory_info =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

    std::array<int64_t, 4> x_shape{1, 1, n_fft / 2 + 1, 2};
    Ort::Value x_tensor = Ort::Value::CreateTensor<float>(
        memory_info, x.data(), x.size(), x_shape.data(), x_shape.size());

    Ort::Value output{nullptr};
    std::tie(output, *next_state) =
        model_.Run(std::move(x_tensor), std::move(state));

    std::vector<float> real(n_fft / 2 + 1);
    std::vector<float> imag(n_fft / 2 + 1);
    const auto *p = output.GetTensorData<float>();
    for (int32_t i = 0; i < n_fft / 2 + 1; ++i) {
      real[i] = p[2 * i];
      imag[i] = p[2 * i + 1];
    }

    return {std::move(real), std::move(imag)};
  }

 private:
  OfflineSpeechDenoiserDpdfNetModel model_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_SPEECH_DENOISER_DPDFNET_IMPL_H_
