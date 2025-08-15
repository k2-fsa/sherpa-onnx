// sherpa-onnx/csrc/offline-tts-zipvoice-impl.h
//
// Copyright (c)  2025  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_OFFLINE_TTS_ZIPVOICE_IMPL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_TTS_ZIPVOICE_IMPL_H_

#include <cmath>
#include <memory>
#include <string>
#include <strstream>
#include <utility>
#include <vector>

#include "kaldi-native-fbank/csrc/mel-computations.h"
#include "kaldi-native-fbank/csrc/stft.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/offline-tts-frontend.h"
#include "sherpa-onnx/csrc/offline-tts-impl.h"
#include "sherpa-onnx/csrc/offline-tts-zipvoice-frontend.h"
#include "sherpa-onnx/csrc/offline-tts-zipvoice-model-config.h"
#include "sherpa-onnx/csrc/offline-tts-zipvoice-model.h"
#include "sherpa-onnx/csrc/onnx-utils.h"
#include "sherpa-onnx/csrc/resample.h"
#include "sherpa-onnx/csrc/vocoder.h"

namespace sherpa_onnx {

class OfflineTtsZipvoiceImpl : public OfflineTtsImpl {
 public:
  explicit OfflineTtsZipvoiceImpl(const OfflineTtsConfig &config)
      : config_(config),
        model_(std::make_unique<OfflineTtsZipvoiceModel>(config.model)),
        vocoder_(Vocoder::Create(config.model)) {
    InitFrontend();
  }

  template <typename Manager>
  OfflineTtsZipvoiceImpl(Manager *mgr, const OfflineTtsConfig &config)
      : config_(config),
        model_(std::make_unique<OfflineTtsZipvoiceModel>(mgr, config.model)),
        vocoder_(Vocoder::Create(mgr, config.model)) {
    InitFrontend(mgr);
  }

  int32_t SampleRate() const override {
    return model_->GetMetaData().sample_rate;
  }

  GeneratedAudio Generate(
      const std::string &text, const std::string &prompt_text,
      const std::vector<float> &prompt_samples, int32_t sample_rate,
      float speed, int num_steps,
      GeneratedAudioCallback callback = nullptr) const override {
    std::vector<TokenIDs> text_token_ids =
        frontend_->ConvertTextToTokenIds(text);

    std::vector<TokenIDs> prompt_token_ids =
        frontend_->ConvertTextToTokenIds(prompt_text);

    if (text_token_ids.empty() ||
        (text_token_ids.size() == 1 && text_token_ids[0].tokens.empty())) {
#if __OHOS__
      SHERPA_ONNX_LOGE("Failed to convert '%{public}s' to token IDs",
                       text.c_str());
#else
      SHERPA_ONNX_LOGE("Failed to convert '%s' to token IDs", text.c_str());
#endif
      return {};
    }

    if (prompt_token_ids.empty() ||
        (prompt_token_ids.size() == 1 && prompt_token_ids[0].tokens.empty())) {
#if __OHOS__
      SHERPA_ONNX_LOGE(
          "Failed to convert prompt text '%{public}s' to token IDs",
          prompt_text.c_str());
#else
      SHERPA_ONNX_LOGE("Failed to convert prompt text '%s' to token IDs",
                       prompt_text.c_str());
#endif
      return {};
    }

    // we assume batch size is 1
    std::vector<int64_t> tokens = text_token_ids[0].tokens;
    std::vector<int64_t> prompt_tokens = prompt_token_ids[0].tokens;

    return Process(tokens, prompt_tokens, prompt_samples, sample_rate, speed,
                   num_steps);
  }

 private:
  template <typename Manager>
  void InitFrontend(Manager *mgr) {
    const auto &meta_data = model_->GetMetaData();
    frontend_ = std::make_unique<OfflineTtsZipvoiceFrontend>(
        mgr, config_.model.zipvoice.tokens, config_.model.zipvoice.data_dir,
        config_.model.zipvoice.pinyin_dict, meta_data, config_.model.debug);
  }

  void InitFrontend() {
    const auto &meta_data = model_->GetMetaData();

    if (meta_data.use_pinyin && config_.model.zipvoice.pinyin_dict.empty()) {
      SHERPA_ONNX_LOGE(
          "Please provide --zipvoice-pinyin-dict for converting Chinese into "
          "pinyin.");
      exit(-1);
    }
    if (meta_data.use_espeak && config_.model.zipvoice.data_dir.empty()) {
      SHERPA_ONNX_LOGE("Please provide --zipvoice-data-dir for espeak-ng.");
      exit(-1);
    }
    frontend_ = std::make_unique<OfflineTtsZipvoiceFrontend>(
        config_.model.zipvoice.tokens, config_.model.zipvoice.data_dir,
        config_.model.zipvoice.pinyin_dict, meta_data, config_.model.debug);
  }

  std::vector<std::vector<float>> ComputeMelSpectrogram(
      const std::vector<float> &_samples, int32_t sample_rate) const {
    // add this in model file
    const auto &meta = model_->GetMetaData();
    if (sample_rate != meta.sample_rate) {
      SHERPA_ONNX_LOGE(
          "Creating a resampler:\n"
          "   in_sample_rate: %d\n"
          "   output_sample_rate: %d\n",
          sample_rate, static_cast<int32_t>(meta.sample_rate));

      float min_freq = std::min<int32_t>(sample_rate, meta.sample_rate);
      float lowpass_cutoff = 0.99 * 0.5 * min_freq;

      int32_t lowpass_filter_width = 6;
      auto resampler = std::make_unique<LinearResample>(
          sample_rate, meta.sample_rate, lowpass_cutoff, lowpass_filter_width);
      std::vector<float> samples;
      resampler->Resample(_samples.data(), _samples.size(), true, &samples);
      return ComputeMelSpectrogram(samples);
    } else {
      // Use the original samples if the sample rate matches
      return ComputeMelSpectrogram(_samples);
    }
  }

  std::vector<std::vector<float>> ComputeMelSpectrogram(
      const std::vector<float> &samples) const {
    // add this in model file
    const auto &meta = model_->GetMetaData();

    int32_t sample_rate = meta.sample_rate;
    int32_t n_fft = meta.n_fft;
    int32_t hop_length = meta.hop_length;
    int32_t win_length = meta.window_length;
    int32_t num_mels = meta.num_mels;

    knf::StftConfig stft_config;
    stft_config.n_fft = n_fft;
    stft_config.hop_length = hop_length;
    stft_config.win_length = win_length;
    stft_config.window_type = "hann";
    stft_config.center = true;

    knf::Stft stft(stft_config);
    auto stft_result = stft.Compute(samples.data(), samples.size());
    int32_t num_frames = stft_result.num_frames;
    int32_t fft_bins = n_fft / 2 + 1;

    knf::FrameExtractionOptions frame_opts;
    frame_opts.samp_freq = sample_rate;
    frame_opts.frame_length_ms = win_length * 1000 / sample_rate;
    frame_opts.frame_shift_ms = hop_length * 1000 / sample_rate;
    frame_opts.window_type = "hanning";

    knf::MelBanksOptions mel_opts;
    mel_opts.num_bins = num_mels;
    mel_opts.low_freq = 0;
    mel_opts.high_freq = sample_rate / 2;
    mel_opts.is_librosa = true;
    mel_opts.use_slaney_mel_scale = false;
    mel_opts.norm = "";

    knf::MelBanks mel_banks(mel_opts, frame_opts, 1.0f);

    std::vector<std::vector<float>> mel_spec;
    mel_spec.reserve(num_frames);

    for (int32_t i = 0; i < num_frames; ++i) {
      std::vector<float> magnitude_spectrum(fft_bins);
      for (int32_t k = 0; k < fft_bins; ++k) {
        float real = stft_result.real[i * fft_bins + k];
        float imag = stft_result.imag[i * fft_bins + k];
        magnitude_spectrum[k] = std::sqrt(real * real + imag * imag);
      }
      std::vector<float> mel_features(num_mels, 0.0f);
      mel_banks.Compute(magnitude_spectrum.data(), mel_features.data());
      for (auto &v : mel_features) {
        v = std::log(v + 1e-10f);
      }
      mel_spec.push_back(std::move(mel_features));
    }
    return mel_spec;
  }

  GeneratedAudio Process(const std::vector<int64_t> &tokens,
                         const std::vector<int64_t> &prompt_tokens,
                         const std::vector<float> &prompt_samples,
                         int32_t sample_rate, float speed,
                         int num_steps) const {
    auto memory_info =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

    std::array<int64_t, 2> tokens_shape = {1,
                                           static_cast<int64_t>(tokens.size())};
    Ort::Value tokens_tensor = Ort::Value::CreateTensor(
        memory_info, const_cast<int64_t *>(tokens.data()), tokens.size(),
        tokens_shape.data(), tokens_shape.size());

    std::array<int64_t, 2> prompt_tokens_shape = {
        1, static_cast<int64_t>(prompt_tokens.size())};
    Ort::Value prompt_tokens_tensor = Ort::Value::CreateTensor(
        memory_info, const_cast<int64_t *>(prompt_tokens.data()),
        prompt_tokens.size(), prompt_tokens_shape.data(),
        prompt_tokens_shape.size());

    float target_rms = config_.model.zipvoice.target_rms;
    float feat_scale = config_.model.zipvoice.feat_scale;

    // Scale prompt_samples
    std::vector<float> prompt_samples_scaled = prompt_samples;
    float prompt_rms = 0.0f;
    double sum_sq = 0.0;
    // Compute RMS of prompt_samples
    for (float s : prompt_samples_scaled) {
      sum_sq += s * s;
    }
    prompt_rms = std::sqrt(sum_sq / prompt_samples_scaled.size());
    if (prompt_rms < target_rms && prompt_rms > 0.0f) {
      float scale = target_rms / static_cast<float>(prompt_rms);
      for (auto &s : prompt_samples_scaled) {
        s *= scale;
      }
    }

    std::vector<std::vector<float>> prompt_features =
        ComputeMelSpectrogram(prompt_samples_scaled, sample_rate);

    const int num_frames = prompt_features.size();
    const int mel_dim = num_frames > 0 ? prompt_features[0].size() : 0;

    if (feat_scale != 1.0f) {
      for (auto &row : prompt_features) {
        for (auto &v : row) {
          v *= feat_scale;
        }
      }
    }

    // Convert the 2D feature matrix into a contiguous 1D array for tensor
    // input Shape: [1, num_frames, mel_dim]
    std::vector<float> prompt_features_flat;
    prompt_features_flat.reserve(num_frames * mel_dim);

    for (int i = 0; i < num_frames; ++i) {
      for (int j = 0; j < mel_dim; ++j) {
        prompt_features_flat.push_back(prompt_features[i][j]);
      }
    }

    std::array<int64_t, 3> shape = {1, num_frames, mel_dim};
    auto prompt_features_tensor = Ort::Value::CreateTensor(
        memory_info, prompt_features_flat.data(), prompt_features_flat.size(),
        shape.data(), shape.size());

    Ort::Value mel =
        model_->Run(std::move(tokens_tensor), std::move(prompt_tokens_tensor),
                    std::move(prompt_features_tensor), speed, num_steps);

    // Assume mel_shape = {1, T, C}
    std::vector<int64_t> mel_shape = mel.GetTensorTypeAndShapeInfo().GetShape();
    int64_t T = mel_shape[1], C = mel_shape[2];

    float *mel_data = mel.GetTensorMutableData<float>();
    std::vector<float> mel_permuted(C * T);

    for (int64_t c = 0; c < C; ++c) {
      for (int64_t t = 0; t < T; ++t) {
        int64_t src_idx = t * C + c;  // src: [T, C] (row major)
        int64_t dst_idx = c * T + t;  // dst: [C, T] (row major)
        mel_permuted[dst_idx] = mel_data[src_idx] / feat_scale;
      }
    }

    std::array<int64_t, 3> new_shape = {1, C, T};
    Ort::Value mel_new = Ort::Value::CreateTensor<float>(
        memory_info, mel_permuted.data(), mel_permuted.size(), new_shape.data(),
        new_shape.size());

    GeneratedAudio ans;
    ans.samples = vocoder_->Run(std::move(mel_new));
    ans.sample_rate = model_->GetMetaData().sample_rate;

    if (prompt_rms < target_rms && target_rms > 0.0f) {
      float scale = prompt_rms / target_rms;
      for (auto &s : ans.samples) {
        s *= scale;
      }
    }
    return ans;
  }

 private:
  OfflineTtsConfig config_;
  std::unique_ptr<OfflineTtsZipvoiceModel> model_;
  std::unique_ptr<Vocoder> vocoder_;
  std::unique_ptr<OfflineTtsFrontend> frontend_;
};

}  // namespace sherpa_onnx
#endif  // SHERPA_ONNX_CSRC_OFFLINE_TTS_ZIPVOICE_IMPL_H_
