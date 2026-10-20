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
#include "sherpa-onnx/csrc/math.h"
#include "sherpa-onnx/csrc/offline-tts-frontend.h"
#include "sherpa-onnx/csrc/offline-tts-impl.h"
#include "sherpa-onnx/csrc/offline-tts-zipvoice-model-config.h"
#include "sherpa-onnx/csrc/offline-tts-zipvoice-model.h"
#include "sherpa-onnx/csrc/onnx-utils.h"
#include "sherpa-onnx/csrc/resample.h"
#include "sherpa-onnx/csrc/symbol-table.h"
#include "sherpa-onnx/csrc/text-utils.h"
#include "sherpa-onnx/csrc/tts-text-normalizer.h"
#include "sherpa-onnx/csrc/vocoder.h"

namespace sherpa_onnx {

class OfflineTtsZipvoiceImpl : public OfflineTtsImpl {
 public:
  explicit OfflineTtsZipvoiceImpl(const OfflineTtsConfig &config)
      : config_(config),
        model_(std::make_unique<OfflineTtsZipvoiceModel>(config.model)),
        vocoder_(Vocoder::Create(config.model)) {
    if (vocoder_ && vocoder_->SampleRate() != model_->GetMetaData().sample_rate) {
      SHERPA_ONNX_LOGE(
          "ERROR: Vocoder sample rate (%d Hz) does not match the ZipVoice "
          "model sample rate (%d Hz).",
          vocoder_->SampleRate(), model_->GetMetaData().sample_rate);
      SHERPA_ONNX_LOGE(
          "Please download the correct vocoder from "
          "https://github.com/k2-fsa/sherpa-onnx/releases/tag/vocoder-models");
      if (model_->GetMetaData().sample_rate == 16000) {
        SHERPA_ONNX_LOGE("For 16kHz models, use vocos-16khz-univ.onnx");
      } else if (model_->GetMetaData().sample_rate == 22050) {
        SHERPA_ONNX_LOGE("For 22050Hz models, use vocos-22khz-univ.onnx");
      } else if (model_->GetMetaData().sample_rate == 24000) {
        SHERPA_ONNX_LOGE("For 24kHz models, use vocos_24khz.onnx");
      }
      SHERPA_ONNX_EXIT(-1);
    }

    if (!config.model.zipvoice.tokens.empty()) {
      auto is = OpenInputFile(config.model.zipvoice.tokens);
      token_str2id_ = ReadTokens(is);
    }

    if (!config.model.zipvoice.lexicon.empty()) {
      std::vector<std::string> files;
      SplitStringToVector(config.model.zipvoice.lexicon, ",", false, &files);
      for (const auto &f : files) {
        auto is = OpenInputFile(f);
        LoadLexicon(is, config.model.debug);
      }
    }

    tn_list_ = LoadTextNormalizers(config.rule_fsts, config.rule_fars,
                                   config.model.debug);

    PostInit();
  }

  template <typename Manager>
  OfflineTtsZipvoiceImpl(Manager *mgr, const OfflineTtsConfig &config)
      : config_(config),
        model_(std::make_unique<OfflineTtsZipvoiceModel>(mgr, config.model)),
        vocoder_(Vocoder::Create(mgr, config.model)) {
    if (vocoder_ && vocoder_->SampleRate() != model_->GetMetaData().sample_rate) {
      SHERPA_ONNX_LOGE(
          "ERROR: Vocoder sample rate (%d Hz) does not match the ZipVoice "
          "model sample rate (%d Hz).",
          vocoder_->SampleRate(), model_->GetMetaData().sample_rate);
      SHERPA_ONNX_LOGE(
          "Please download the correct vocoder from "
          "https://github.com/k2-fsa/sherpa-onnx/releases/tag/vocoder-models");
      if (model_->GetMetaData().sample_rate == 16000) {
        SHERPA_ONNX_LOGE("For 16kHz models, use vocos-16khz-univ.onnx");
      } else if (model_->GetMetaData().sample_rate == 22050) {
        SHERPA_ONNX_LOGE("For 22050Hz models, use vocos-22khz-univ.onnx");
      } else if (model_->GetMetaData().sample_rate == 24000) {
        SHERPA_ONNX_LOGE("For 24kHz models, use vocos_24khz.onnx");
      }
      SHERPA_ONNX_EXIT(-1);
    }

    if (!config.model.zipvoice.tokens.empty()) {
      auto buf = ReadFile(mgr, config.model.zipvoice.tokens);
      std::istringstream is(std::string(buf.data(), buf.size()));
      token_str2id_ = ReadTokens(is);
    }

    if (!config.model.zipvoice.lexicon.empty()) {
      std::vector<std::string> files;
      SplitStringToVector(config.model.zipvoice.lexicon, ",", false, &files);
      for (const auto &f : files) {
        auto buf = ReadFile(mgr, f);
        std::istringstream is(std::string(buf.data(), buf.size()));
        LoadLexicon(is, config.model.debug);
      }
    }

    tn_list_ = LoadTextNormalizers(mgr, config.rule_fsts, config.rule_fars,
                                   config.model.debug);

    PostInit();
  }

  int32_t SampleRate() const override {
    return model_->GetMetaData().sample_rate;
  }

  /**
   *
   * Supported options in GenerationConfig:
   *   - speed: Speech speed factor (default: 1.0)
   *   - silence_scale: Scale applied to pauses in the generated audio
   *   - reference_audio: Mono float32 audio samples for zero-shot cloning
   *   - reference_sample_rate: Sample rate of reference_audio
   *   - reference_text: Transcript of reference_audio
   *   - tokens: Vector of sentences, each a vector of token strings.
   *     tokens[0] is used for the reference text, tokens[1..] for generated text.
   *
   * Supported extra parameters:
   *
   *  - debug, int, default from model config
   *  - min_words_in_sentence, int, default 5.
   *    Merge adjacent sentences if the number of words is less than this.
   *    Each CJK character counts as one word; English words are space-separated.
   *  - max_words_in_sentence, int, default 20.
   *    Split a sentence into chunks if the number of words exceeds this.
   *    Splits at punctuation or space boundaries.
   *  - num_steps, int, default 4.
   *    Number of flow-matching denoising steps.
   *  - feat_scale, float, default from model config.
   *    Prompt mel log scaling factor.
   *  - t_shift, float, default from model config.
   *    Timestep shift used by the decoder schedule.
   *  - target_rms, float, default from model config.
   *    Prompt RMS normalization target.
   *  - guidance_scale, float, default from model config.
   *    Classifier-free guidance scale for the decoder.
   */
  GeneratedAudio Generate(
      const std::string &text, const GenerationConfig &config,
      GeneratedAudioCallback callback = nullptr) const override {
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

    // Tokenize reference text
    std::vector<int64_t> prompt_tokens;
    if (!config.phoneme_codepoints.empty()) {
      SHERPA_ONNX_LOGE(
          "GenerationConfig.phoneme_codepoints is not supported by ZipVoice "
          "models and is ignored. Use lexicon or GenerationConfig.tokens "
          "instead.");
    }
    if (!config.tokens.empty()) {
      // tokens path: use first sentence as prompt tokens
      for (const auto &tok : config.tokens[0]) {
        auto it = token_str2id_.find(tok);
        if (it != token_str2id_.end()) {
          prompt_tokens.push_back(it->second);
        }
      }
    } else {
      prompt_tokens = TokenizeText(config.reference_text, config_.model.debug);
    }
    if (prompt_tokens.empty()) {
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

    std::vector<float> prompt_features = ComputePromptFeatures(
        config.reference_audio, config.reference_sample_rate, feat_scale,
        target_rms);
    if (prompt_features.empty()) {
      SHERPA_ONNX_LOGE("No frames extracted from the prompt audio");
      return {};
    }

    GeneratedAudio result;
    result.sample_rate = SampleRate();

    if (!config.tokens.empty()) {
      // tokens path: use token strings directly for generated text
      // config.tokens[0] was used for prompt, config.tokens[1..] for generated

      // Flatten all generated tokens
      std::vector<std::string> all_tokens;
      for (size_t si = 1; si < config.tokens.size(); ++si) {
        for (const auto &tok : config.tokens[si]) {
          all_tokens.push_back(tok);
        }
      }

      // Split at punctuation tokens into sub-sentences
      auto IsPunctToken = [](const std::string &t) {
        return t == "," || t == "." || t == "!" || t == "?" || t == ";" ||
               t == ":";
      };

      std::vector<std::vector<std::string>> token_sentences;
      std::vector<std::string> current;
      for (const auto &tok : all_tokens) {
        current.push_back(tok);
        if (IsPunctToken(tok)) {
          token_sentences.push_back(std::move(current));
          current.clear();
        }
      }
      if (!current.empty()) {
        token_sentences.push_back(std::move(current));
      }

      // Merge short token sentences
      int32_t min_words =
          config.GetExtraInt("min_words_in_sentence", 5);
      std::vector<std::vector<std::string>> merged;
      std::vector<std::string> buffer;
      for (auto &ts : token_sentences) {
        buffer.insert(buffer.end(), ts.begin(), ts.end());
        if (static_cast<int32_t>(buffer.size()) >= min_words) {
          merged.push_back(std::move(buffer));
          buffer.clear();
        }
      }
      if (!buffer.empty()) {
        if (!merged.empty()) {
          merged.back().insert(merged.back().end(), buffer.begin(),
                               buffer.end());
        } else {
          merged.push_back(std::move(buffer));
        }
      }

      if (config_.model.debug) {
#if __OHOS__
        SHERPA_ONNX_LOGE("Reference text: %{public}s",
                         config.reference_text.c_str());
        SHERPA_ONNX_LOGE("Using tokens path: %{public}d sentence(s)",
                         static_cast<int32_t>(merged.size()));
#else
        SHERPA_ONNX_LOGE("Reference text: %s",
                         config.reference_text.c_str());
        SHERPA_ONNX_LOGE("Using tokens path: %d sentence(s)",
                         static_cast<int32_t>(merged.size()));
#endif
      }

      // Convert token strings to IDs and process each sentence
      int32_t sentence_idx = 0;
      for (const auto &sentence : merged) {
        std::vector<int64_t> tokens;
        std::vector<std::string> id_strs;
        for (const auto &tok : sentence) {
          auto it = token_str2id_.find(tok);
          if (it != token_str2id_.end()) {
            tokens.push_back(it->second);
            id_strs.push_back(tok + "(" + std::to_string(it->second) + ")");
          } else {
            if (config_.model.debug) {
              SHERPA_ONNX_LOGE("Skip unknown token: '%s'", tok.c_str());
            }
          }
        }

        // Merge single-token sentence into previous
        if (tokens.size() == 1 && !result.samples.empty()) {
          // Skip — don't process single-token sentences alone
          continue;
        }

        if (tokens.empty()) continue;

        if (config_.model.debug) {
          std::ostringstream os;
          os << "Sentence " << sentence_idx << " tokens: [";
          for (size_t j = 0; j < id_strs.size(); ++j) {
            if (j > 0) os << ", ";
            os << id_strs[j];
          }
          os << "]";
#if __OHOS__
          SHERPA_ONNX_LOGE("%{public}s", os.str().c_str());
#else
          SHERPA_ONNX_LOGE("%s", os.str().c_str());
#endif
          ++sentence_idx;
        }

        GeneratedAudio cur = Process(tokens, prompt_tokens, prompt_features,
                                     speed, num_steps, feat_scale, t_shift,
                                     guidance_scale);
        if (cur.samples.empty()) continue;
        result.samples.insert(result.samples.end(), cur.samples.begin(),
                              cur.samples.end());
      }
    } else {
      // Text path: convert punctuations, split into sentences, merge short,
      // split long, tokenize each.
      if (config_.model.debug) {
#if __OHOS__
        SHERPA_ONNX_LOGE("Raw text: %{public}s", text.c_str());
#else
        SHERPA_ONNX_LOGE("Raw text: %s", text.c_str());
#endif
      }

      std::string normalized = ReplacePunctuations(text);

      auto sentences = SplitByAllPunctuation(normalized);
      if (sentences.empty()) {
        return {};
      }

      int32_t min_words =
          config.GetExtraInt("min_words_in_sentence", 5);
      int32_t max_words =
          config.GetExtraInt("max_words_in_sentence", 20);

      sentences = MergeShortSentencesByWords(sentences, min_words);

      std::vector<std::string> final_chunks;
      for (const auto &s : sentences) {
        auto pieces = SplitLongSentenceByWords(s, max_words);
        final_chunks.insert(final_chunks.end(), pieces.begin(), pieces.end());
      }

      // Merge punctuation-only chunks into previous
      std::vector<std::string> merged_chunks;
      for (const auto &s : final_chunks) {
        std::u32string u32 = Utf8ToUtf32(Trim(s));
        bool all_punct = !u32.empty();
        for (char32_t c : u32) {
          if (c != U',' && c != U'.' && c != U'!' && c != U'?' &&
              c != U';' && c != U':' && c != U' ') {
            all_punct = false;
            break;
          }
        }
        if (all_punct && !merged_chunks.empty()) {
          merged_chunks.back() += Trim(s);
        } else {
          merged_chunks.push_back(s);
        }
      }
      sentences = std::move(merged_chunks);

      if (config_.model.debug) {
        SHERPA_ONNX_LOGE("After split/merge: %d sentences",
                         static_cast<int32_t>(sentences.size()));
        for (int32_t i = 0; i < static_cast<int32_t>(sentences.size()); ++i) {
#if __OHOS__
          SHERPA_ONNX_LOGE("  sentence %d: '%{public}s'", i,
                           sentences[i].c_str());
#else
          SHERPA_ONNX_LOGE("  sentence %d: '%s'", i, sentences[i].c_str());
#endif
        }
      }

      const int32_t total = static_cast<int32_t>(sentences.size());

      for (int32_t i = 0; i < total; ++i) {
        if (config_.model.debug) {
#if __OHOS__
          SHERPA_ONNX_LOGE("Processing %{public}d/%{public}d: %{public}s",
                           i + 1, total, sentences[i].c_str());
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
    }  // end else (text path)

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
    // No longer using MatchaTtsLexicon (espeak-ng removed).
    // Lexicon-based tokenization is handled by LoadLexicon + TokenizeText.
  }

  void InitFrontend() {
    // No longer using MatchaTtsLexicon (espeak-ng removed).
    // Lexicon-based tokenization is handled by LoadLexicon + TokenizeText.
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
    std::vector<int64_t> tokens = TokenizeText(text, config_.model.debug);

    if (tokens.empty()) {
#if __OHOS__
      SHERPA_ONNX_LOGE("Failed to convert '%{public}s' to token IDs",
                       text.c_str());
#else
      SHERPA_ONNX_LOGE("Failed to convert '%s' to token IDs", text.c_str());
#endif
      return {};
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
    if (!mel) {
      return {};
    }

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

  void LoadLexicon(std::istream &is, bool debug) {
    auto entries = ParseLexiconFile(is, &max_lexicon_phrase_len_);
    for (auto &e : entries) {
      lexicon_str_[std::move(e.key)] = std::move(e.phonemes);
    }
    if (debug) {
      SHERPA_ONNX_LOGE("Loaded lexicon: %d entries, max phrase len %d",
                       static_cast<int32_t>(lexicon_str_.size()),
                       max_lexicon_phrase_len_);
    }
  }

  std::vector<int64_t> TokenizeText(const std::string &text,
                                    bool debug) const {
    std::vector<int64_t> ids;
    std::string normalized = ReplacePunctuations(text);

    if (debug) {
#if __OHOS__
      SHERPA_ONNX_LOGE("TokenizeText: text='%{public}s'", normalized.c_str());
#else
      SHERPA_ONNX_LOGE("TokenizeText: text='%s'", normalized.c_str());
#endif
    }

    std::vector<std::string> words = SplitUtf8(normalized);

    int32_t i = 0;
    int32_t n = static_cast<int32_t>(words.size());
    while (i < n) {
      if (words[i] == " ") {
        auto it = token_str2id_.find(" ");
        if (it != token_str2id_.end()) ids.push_back(it->second);
        ++i;
        continue;
      }
      if (IsPunctWord(words[i])) {
        auto it = token_str2id_.find(words[i]);
        if (it != token_str2id_.end()) {
          ids.push_back(it->second);
          if (debug) {
            SHERPA_ONNX_LOGE("Punctuation: '%s' -> %d", words[i].c_str(),
                             it->second);
          }
        }
        ++i;
        continue;
      }
      bool found = false;
      int32_t max_try = std::min(max_lexicon_phrase_len_, n - i);
      for (int32_t len = max_try; len >= 1; --len) {
        std::string phrase;
        for (int32_t j = i; j < i + len; ++j) {
          bool prev_cjk = !phrase.empty() && IsCJK(Utf8ToUtf32(phrase).back());
          bool curr_cjk = !words[j].empty() && IsCJK(Utf8ToUtf32(words[j]).front());
          if (j > i && !(prev_cjk && curr_cjk)) phrase.push_back(' ');
          phrase += ToLowerCase(words[j]);
        }
        auto it = lexicon_str_.find(phrase);
        if (it != lexicon_str_.end()) {
          if (debug) {
            std::string phones_str;
            for (const auto &p : it->second) phones_str += p + " ";
            SHERPA_ONNX_LOGE("Lexicon matched: '%s' -> '%s'", phrase.c_str(),
                             phones_str.c_str());
          }
          auto tok_ids = ConvertPhonemeStringsToIds(it->second);
          ids.insert(ids.end(), tok_ids.begin(), tok_ids.end());
          i += len;
          found = true;
          break;
        }
      }
      if (!found) {
        if (ContainsCJK(words[i])) {
          for (const auto &ch : SplitUtf8(words[i])) {
            std::string ch_lower = ToLowerCase(ch);
            auto it = lexicon_str_.find(ch_lower);
            if (it != lexicon_str_.end()) {
              if (debug) {
                std::string phones_str;
                for (const auto &p : it->second) phones_str += p + " ";
                SHERPA_ONNX_LOGE("Lexicon matched: '%s' -> '%s'",
                                 ch_lower.c_str(), phones_str.c_str());
              }
              auto tok_ids = ConvertPhonemeStringsToIds(it->second);
              ids.insert(ids.end(), tok_ids.begin(), tok_ids.end());
            } else if (debug) {
              SHERPA_ONNX_LOGE("OOV character skipped: '%s'", ch.c_str());
            }
          }
        } else {
          // The word is not in the lexicon. If it contains '-', split it and
          // look up each part separately, e.g. "two-thirds" -> "two" +
          // "thirds".
          std::vector<std::string> parts = SplitStringAndTrim(words[i], '-');
          if (words[i].find('-') != std::string::npos && !parts.empty()) {
            for (const auto &part : parts) {
              std::string part_lower = ToLowerCase(part);
              auto it2 = lexicon_str_.find(part_lower);
              if (it2 != lexicon_str_.end()) {
                if (debug) {
                  std::string phones_str;
                  for (const auto &p : it2->second) phones_str += p + " ";
                  SHERPA_ONNX_LOGE("Lexicon matched: '%s' -> '%s'",
                                   part_lower.c_str(), phones_str.c_str());
                }
                auto tok_ids = ConvertPhonemeStringsToIds(it2->second);
                ids.insert(ids.end(), tok_ids.begin(), tok_ids.end());
              } else if (debug) {
                SHERPA_ONNX_LOGE("OOV word skipped: '%s'", part.c_str());
              }
            }
          } else if (debug) {
            SHERPA_ONNX_LOGE("OOV word skipped: '%s'", words[i].c_str());
          }
        }
        ++i;
      }
    }

    if (debug) {
      std::ostringstream os;
      os << "Tokens (" << ids.size() << "): [";
      for (size_t j = 0; j < ids.size(); ++j) {
        if (j > 0) os << ", ";
        // Look up token string
        std::string tok_str = "?";
        for (const auto &kv : token_str2id_) {
          if (kv.second == ids[j]) {
            tok_str = kv.first;
            break;
          }
        }
        os << tok_str << "(" << ids[j] << ")";
      }
      os << "]";
#if __OHOS__
      SHERPA_ONNX_LOGE("%{public}s", os.str().c_str());
#else
      SHERPA_ONNX_LOGE("%s", os.str().c_str());
#endif
    }

    return ids;
  }

  std::vector<int64_t> ConvertPhonemeStringsToIds(
      const std::vector<std::string> &phones) const {
    std::vector<int64_t> ans;
    for (const auto &p : phones) {
      auto it = token_str2id_.find(p);
      if (it != token_str2id_.end()) ans.push_back(it->second);
    }
    return ans;
  }

  static std::string ReplacePunctuations(const std::string &s) {
    static const std::vector<std::pair<std::string, std::string>> r = {
        {"，", ","}, {"、", ","}, {"；", ";"}, {"：", ","}, {":", ","},
        {"。", "."}, {"？", "?"}, {"！", "!"}, {"…", "..."},
    };
    std::string result = s;
    for (const auto &p : r) {
      size_t pos = 0;
      while ((pos = result.find(p.first, pos)) != std::string::npos) {
        result.replace(pos, p.first.size(), p.second);
        pos += p.second.size();
      }
    }
    return result;
  }

  static bool IsPunctWord(const std::string &s) {
    if (s.empty()) return false;
    static const std::string puncts = ",.;:!?";
    return s.size() == 1 && puncts.find(s[0]) != std::string::npos;
  }

 private:
  OfflineTtsConfig config_;
  std::unique_ptr<OfflineTtsZipvoiceModel> model_;
  std::unique_ptr<Vocoder> vocoder_;
  std::vector<std::unique_ptr<kaldifst::TextNormalizer>> tn_list_;
  std::unordered_map<std::string, int32_t> token_str2id_;
  std::unordered_map<std::string, std::vector<std::string>> lexicon_str_;
  int32_t max_lexicon_phrase_len_ = 1;
  std::unique_ptr<knf::MelBanks> mel_banks_;
};

}  // namespace sherpa_onnx
#endif  // SHERPA_ONNX_CSRC_OFFLINE_TTS_ZIPVOICE_IMPL_H_
