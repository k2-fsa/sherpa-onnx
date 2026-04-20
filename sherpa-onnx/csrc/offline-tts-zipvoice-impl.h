// sherpa-onnx/csrc/offline-tts-zipvoice-impl.h
//
// Copyright (c)  2025  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_OFFLINE_TTS_ZIPVOICE_IMPL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_TTS_ZIPVOICE_IMPL_H_

#include <algorithm>
#include <cmath>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "kaldi-native-fbank/csrc/mel-computations.h"
#include "kaldi-native-fbank/csrc/stft.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/matcha-tts-lexicon.h"
#include "sherpa-onnx/csrc/math.h"
#include "sherpa-onnx/csrc/offline-tts-frontend.h"
#include "sherpa-onnx/csrc/offline-tts-impl.h"
#include "sherpa-onnx/csrc/offline-tts-zipvoice-model-config.h"
#include "sherpa-onnx/csrc/offline-tts-zipvoice-model.h"
#include "sherpa-onnx/csrc/onnx-utils.h"
#include "sherpa-onnx/csrc/resample.h"
#include "sherpa-onnx/csrc/text-utils.h"
#include "sherpa-onnx/csrc/vocoder.h"

namespace sherpa_onnx {

class OfflineTtsZipvoiceImpl : public OfflineTtsImpl {
 public:
  explicit OfflineTtsZipvoiceImpl(const OfflineTtsConfig &config)
      : config_(config),
        model_(std::make_unique<OfflineTtsZipvoiceModel>(config.model)),
        vocoder_(Vocoder::Create(config.model)) {
    InitFrontend();

    PostInit();
  }

  template <typename Manager>
  OfflineTtsZipvoiceImpl(Manager *mgr, const OfflineTtsConfig &config)
      : config_(config),
        model_(std::make_unique<OfflineTtsZipvoiceModel>(mgr, config.model)),
        vocoder_(Vocoder::Create(mgr, config.model)) {
    InitFrontend(mgr);

    PostInit();
  }

  int32_t SampleRate() const override {
    return model_->GetMetaData().sample_rate;
  }

  GeneratedAudio Generate(
      const std::string &text, const GenerationConfig &config,
      GeneratedAudioCallback callback = nullptr) const override {
    // Supported extra options in config.extra:
    //   - "speed" (float): Speech speed factor (default: 1.0)
    //   - "num_steps" (int): Number of flow-matching steps (default: 4)
    //   - "max_char_in_sentence" (int): Max characters per chunk (default: 200)
    //   - "min_char_in_sentence" (int): Merge shorter chunks until this size
    //     (default: 30)
    //   - "feat_scale" (float): Prompt mel log scaling factor (default:
    //     config.model.zipvoice.feat_scale)
    //   - "t_shift" (float): Timestep shift used by the decoder schedule
    //     (default: config.model.zipvoice.t_shift)
    //   - "target_rms" (float): Prompt RMS normalization target (default:
    //     config.model.zipvoice.target_rms)
    //   - "guidance_scale" (float): Classifier-free guidance scale for the
    //     decoder (default: config.model.zipvoice.guidance_scale)
    if (config_.model.debug) {
      SHERPA_ONNX_LOGE("%s", config.ToString().c_str());
    }

    if (config.reference_sample_rate <= 0) {
      SHERPA_ONNX_LOGE("reference_sample_rate %d is invalid.",
                       config.reference_sample_rate);
      return {};
    }

    if (config.reference_audio.empty()) {
      SHERPA_ONNX_LOGE("reference_audio is empty.");
      return {};
    }

    if (config.reference_text.empty()) {
      SHERPA_ONNX_LOGE("reference_text is empty.");
      return {};
    }

    float speed =
        config.GetExtraFloat("speed", config.speed > 0 ? config.speed : 1.0f);
    if (speed <= 0) {
      SHERPA_ONNX_LOGE("Speed must be > 0. Given: %f", speed);
      return {};
    }

    int32_t num_steps = config.GetExtraInt(
        "num_steps", config.num_steps > 0 ? config.num_steps : 4);
    if (num_steps <= 0) {
      SHERPA_ONNX_LOGE("Num steps must be > 0. Given: %d", num_steps);
      return {};
    }

    float feat_scale =
        config.GetExtraFloat("feat_scale", config_.model.zipvoice.feat_scale);
    if (feat_scale <= 0) {
      SHERPA_ONNX_LOGE("feat_scale must be > 0. Given: %f", feat_scale);
      return {};
    }

    float t_shift =
        config.GetExtraFloat("t_shift", config_.model.zipvoice.t_shift);
    if (t_shift < 0) {
      SHERPA_ONNX_LOGE("t_shift must be >= 0. Given: %f", t_shift);
      return {};
    }

    float target_rms =
        config.GetExtraFloat("target_rms", config_.model.zipvoice.target_rms);
    if (target_rms <= 0) {
      SHERPA_ONNX_LOGE("target_rms must be > 0. Given: %f", target_rms);
      return {};
    }

    float guidance_scale = config.GetExtraFloat(
        "guidance_scale", config_.model.zipvoice.guidance_scale);
    if (guidance_scale <= 0) {
      SHERPA_ONNX_LOGE("guidance_scale must be > 0. Given: %f", guidance_scale);
      return {};
    }

    std::vector<TokenIDs> prompt_token_ids =
        frontend_->ConvertTextToTokenIds(config.reference_text);
    if (prompt_token_ids.empty() ||
        (prompt_token_ids.size() == 1 && prompt_token_ids[0].tokens.empty())) {
#if __OHOS__
      SHERPA_ONNX_LOGE(
          "Failed to convert prompt text '%{public}s' to token IDs",
          config.reference_text.c_str());
#else
      SHERPA_ONNX_LOGE("Failed to convert prompt text '%s' to token IDs",
                       config.reference_text.c_str());
#endif
      return {};
    }

    std::vector<int64_t> prompt_tokens;
    for (const auto &t : prompt_token_ids) {
      prompt_tokens.insert(prompt_tokens.end(), t.tokens.begin(),
                           t.tokens.end());
    }

    std::vector<float> prompt_features = ComputePromptFeatures(
        config.reference_audio, config.reference_sample_rate, feat_scale,
        target_rms);
    if (prompt_features.empty()) {
      SHERPA_ONNX_LOGE("No frames extracted from the prompt audio");
      return {};
    }

    auto sentences = SplitByPunctuation(text);
    if (sentences.empty()) {
      return {};
    }

    int32_t max_char_in_sentence =
        config.GetExtraInt("max_char_in_sentence", 200);
    int32_t min_char_in_sentence =
        config.GetExtraInt("min_char_in_sentence", 30);

    if (max_char_in_sentence <= 0) {
      SHERPA_ONNX_LOGE("max_char_in_sentence must be > 0. Given: %d",
                       max_char_in_sentence);
      return {};
    }

    if (min_char_in_sentence <= 0) {
      SHERPA_ONNX_LOGE("min_char_in_sentence must be > 0. Given: %d",
                       min_char_in_sentence);
      return {};
    }

    sentences = MergeShortSentences(sentences, min_char_in_sentence);

    std::vector<std::string> final_chunks;
    for (const auto &s : sentences) {
      auto pieces = SplitLongSentence(s, max_char_in_sentence);
      final_chunks.insert(final_chunks.end(), pieces.begin(), pieces.end());
    }

    sentences = std::move(final_chunks);
    if (sentences.empty()) {
      return {};
    }

    GeneratedAudio result;
    result.sample_rate = SampleRate();

    const int32_t total = static_cast<int32_t>(sentences.size());

    for (int32_t i = 0; i < total; ++i) {
      if (config_.model.debug) {
#if __OHOS__
        SHERPA_ONNX_LOGE("Processing %{public}d/%{public}d: %{public}s", i + 1,
                         total, sentences[i].c_str());
#else
        SHERPA_ONNX_LOGE("Processing %d/%d: %s", i + 1, total,
                         sentences[i].c_str());
#endif
      }

      GeneratedAudio cur = GenerateChunk(
          sentences[i], prompt_tokens, prompt_features, speed, num_steps,
          feat_scale, t_shift, guidance_scale);

      if (cur.samples.empty()) {
        continue;
      }

      result.samples.insert(result.samples.end(), cur.samples.begin(),
                            cur.samples.end());

      if (callback) {
        if (!callback(cur.samples.data(),
                      static_cast<int32_t>(cur.samples.size()),
                      (i + 1) * 1.0f / total)) {
          break;
        }
      }
    }

    if (config.silence_scale != 1) {
      result = result.ScaleSilence(config.silence_scale);
    }

    return result;
  }

  GeneratedAudio Generate(
      const std::string &text, const std::string &prompt_text,
      const std::vector<float> &prompt_samples, int32_t sample_rate,
      float speed, int32_t num_steps,
      GeneratedAudioCallback callback = nullptr) const override {
    GenerationConfig config;
    config.speed = speed;
    config.num_steps = num_steps;
    config.reference_text = prompt_text;
    config.reference_audio = prompt_samples;
    config.reference_sample_rate = sample_rate;
    return Generate(text, config, std::move(callback));
  }

 private:
  void PostInit() { InitMelBanks(); }

  void InitMelBanks() {
    const auto &meta = model_->GetMetaData();
    int32_t sample_rate = meta.sample_rate;
    int32_t n_fft = meta.n_fft;
    int32_t hop_length = meta.hop_length;
    int32_t win_length = meta.window_length;
    int32_t num_mels = meta.num_mels;

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

    mel_banks_ = std::make_unique<knf::MelBanks>(mel_opts, frame_opts, 1.0f);
  }

  template <typename Manager>
  void InitFrontend(Manager *mgr) {
    frontend_ = std::make_unique<MatchaTtsLexicon>(
        mgr, config_.model.zipvoice.lexicon, config_.model.zipvoice.tokens,
        config_.model.zipvoice.data_dir, config_.model.debug, true);
  }

  void InitFrontend() {
    frontend_ = std::make_unique<MatchaTtsLexicon>(
        config_.model.zipvoice.lexicon, config_.model.zipvoice.tokens,
        config_.model.zipvoice.data_dir, config_.model.debug, true);
  }

  void ComputeMelSpectrogram(const std::vector<float> &_samples,
                             int32_t sample_rate, float feat_scale,
                             std::vector<float> *prompt_features) const {
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
      ComputeMelSpectrogram(samples, feat_scale, prompt_features);
      return;
    }

    ComputeMelSpectrogram(_samples, feat_scale, prompt_features);
  }

  void ComputeMelSpectrogram(const std::vector<float> &samples,
                             float feat_scale,
                             std::vector<float> *prompt_features) const {
    const auto &meta = model_->GetMetaData();

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

    prompt_features->resize(num_frames * num_mels);
    float *p = prompt_features->data();

    std::vector<float> magnitude_spectrum(fft_bins);

    for (int32_t i = 0; i < num_frames; ++i, p += num_mels) {
      for (int32_t k = 0; k < fft_bins; ++k) {
        float real = stft_result.real[i * fft_bins + k];
        float imag = stft_result.imag[i * fft_bins + k];
        magnitude_spectrum[k] = std::sqrt(real * real + imag * imag);
      }

      mel_banks_->Compute(magnitude_spectrum.data(), p);

      for (int32_t j = 0; j < num_mels; ++j) {
        p[j] = std::log(p[j] + 1e-10f) * feat_scale;
      }
    }
  }

  GeneratedAudio GenerateChunk(const std::string &text,
                               const std::vector<int64_t> &prompt_tokens,
                               const std::vector<float> &prompt_features,
                               float speed, int32_t num_steps, float feat_scale,
                               float t_shift, float guidance_scale) const {
    std::vector<TokenIDs> text_token_ids =
        frontend_->ConvertTextToTokenIds(text);

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

    std::vector<int64_t> tokens;
    for (const auto &t : text_token_ids) {
      tokens.insert(tokens.end(), t.tokens.begin(), t.tokens.end());
    }

    return Process(tokens, prompt_tokens, prompt_features, speed, num_steps,
                   feat_scale, t_shift, guidance_scale);
  }

  std::vector<float> ComputePromptFeatures(
      const std::vector<float> &prompt_samples, int32_t sample_rate,
      float feat_scale, float target_rms) const {
    std::vector<float> prompt_samples_scaled = prompt_samples;
    double prompt_rms = 0.0;
    double sum_sq = 0.0;
    for (float s : prompt_samples_scaled) {
      sum_sq += s * s;
    }
    prompt_rms = std::sqrt(sum_sq / prompt_samples_scaled.size());
    if (prompt_rms < target_rms && prompt_rms > 0.0f) {
      float scale = target_rms / prompt_rms;
      for (auto &s : prompt_samples_scaled) {
        s *= scale;
      }
    }

    std::vector<float> prompt_features;
    ComputeMelSpectrogram(prompt_samples_scaled, sample_rate, feat_scale,
                          &prompt_features);

    return prompt_features;
  }

  GeneratedAudio Process(const std::vector<int64_t> &tokens,
                         const std::vector<int64_t> &prompt_tokens,
                         const std::vector<float> &prompt_features, float speed,
                         int32_t num_steps, float feat_scale, float t_shift,
                         float guidance_scale) const {
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

    int32_t mel_dim = model_->GetMetaData().num_mels;

    int32_t num_frames = prompt_features.size() / mel_dim;

    std::array<int64_t, 3> shape = {1, num_frames, mel_dim};
    auto prompt_features_tensor = Ort::Value::CreateTensor(
        memory_info, const_cast<float *>(prompt_features.data()),
        prompt_features.size(), shape.data(), shape.size());

    Ort::Value mel =
        model_->Run(std::move(tokens_tensor), std::move(prompt_tokens_tensor),
                    std::move(prompt_features_tensor), speed, num_steps,
                    t_shift, guidance_scale);

    // Assume mel_shape = {1, T, C}
    std::vector<int64_t> mel_shape = mel.GetTensorTypeAndShapeInfo().GetShape();
    int64_t T = mel_shape[1];
    int64_t C = mel_shape[2];

    const float *mel_data = mel.GetTensorData<float>();

    float inv_feat_scale = 1 / feat_scale;

    // mel_permuted is (C, T)
    std::vector<float> mel_permuted = Transpose(mel_data, T, C);

    Scale(mel_permuted.data(), inv_feat_scale, mel_permuted.size(),
          mel_permuted.data());

    std::array<int64_t, 3> new_shape = {1, C, T};
    Ort::Value mel_new = Ort::Value::CreateTensor<float>(
        memory_info, mel_permuted.data(), mel_permuted.size(), new_shape.data(),
        new_shape.size());

    GeneratedAudio ans;
    ans.samples = vocoder_->Run(std::move(mel_new));
    ans.sample_rate = model_->GetMetaData().sample_rate;
    return ans;
  }

 private:
  OfflineTtsConfig config_;
  std::unique_ptr<OfflineTtsZipvoiceModel> model_;
  std::unique_ptr<Vocoder> vocoder_;
  std::unique_ptr<OfflineTtsFrontend> frontend_;

  std::unique_ptr<knf::MelBanks> mel_banks_;
};

}  // namespace sherpa_onnx
#endif  // SHERPA_ONNX_CSRC_OFFLINE_TTS_ZIPVOICE_IMPL_H_
