// sherpa-onnx/csrc/offline-tts-kokoro-impl.h
//
// Copyright (c)  2025  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_OFFLINE_TTS_KOKORO_IMPL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_TTS_KOKORO_IMPL_H_

#include <iomanip>
#include <ios>
#include <memory>
#include <string>
#include <sstream>
#include <utility>
#include <vector>

#include "fst/extensions/far/far.h"
#include "kaldifst/csrc/kaldi-fst-io.h"
#include "kaldifst/csrc/text-normalizer.h"
#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/kokoro-multi-lang-lexicon.h"
#include "sherpa-onnx/csrc/lexicon.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/offline-tts-frontend.h"
#include "sherpa-onnx/csrc/offline-tts-impl.h"
#include "sherpa-onnx/csrc/offline-tts-kokoro-model.h"
#include "sherpa-onnx/csrc/piper-phonemize-lexicon.h"
#include "sherpa-onnx/csrc/text-utils.h"

namespace sherpa_onnx {

class OfflineTtsKokoroImpl : public OfflineTtsImpl {
 public:
  explicit OfflineTtsKokoroImpl(const OfflineTtsConfig &config)
      : config_(config),
        model_(std::make_unique<OfflineTtsKokoroModel>(config.model)) {
    InitFrontend();

    if (!config.rule_fsts.empty()) {
      std::vector<std::string> files;
      SplitStringToVector(config.rule_fsts, ",", false, &files);
      tn_list_.reserve(files.size());
      for (const auto &f : files) {
        if (config.model.debug) {
#if __OHOS__
          SHERPA_ONNX_LOGE("rule fst: %{public}s", f.c_str());
#else
          SHERPA_ONNX_LOGE("rule fst: %s", f.c_str());
#endif
        }
        tn_list_.push_back(std::make_unique<kaldifst::TextNormalizer>(f));
      }
    }

    if (!config.rule_fars.empty()) {
      if (config.model.debug) {
        SHERPA_ONNX_LOGE("Loading FST archives");
      }
      std::vector<std::string> files;
      SplitStringToVector(config.rule_fars, ",", false, &files);

      tn_list_.reserve(files.size() + tn_list_.size());

      for (const auto &f : files) {
        if (config.model.debug) {
#if __OHOS__
          SHERPA_ONNX_LOGE("rule far: %{public}s", f.c_str());
#else
          SHERPA_ONNX_LOGE("rule far: %s", f.c_str());
#endif
        }
        std::unique_ptr<fst::FarReader<fst::StdArc>> reader(
            fst::FarReader<fst::StdArc>::Open(f));
        for (; !reader->Done(); reader->Next()) {
          std::unique_ptr<fst::StdConstFst> r(
              fst::CastOrConvertToConstFst(reader->GetFst()->Copy()));

          tn_list_.push_back(
              std::make_unique<kaldifst::TextNormalizer>(std::move(r)));
        }
      }

      if (config.model.debug) {
        SHERPA_ONNX_LOGE("FST archives loaded!");
      }
    }
  }

  template <typename Manager>
  OfflineTtsKokoroImpl(Manager *mgr, const OfflineTtsConfig &config)
      : config_(config),
        model_(std::make_unique<OfflineTtsKokoroModel>(mgr, config.model)) {
    InitFrontend(mgr);

    if (!config.rule_fsts.empty()) {
      std::vector<std::string> files;
      SplitStringToVector(config.rule_fsts, ",", false, &files);
      tn_list_.reserve(files.size());
      for (const auto &f : files) {
        if (config.model.debug) {
#if __OHOS__
          SHERPA_ONNX_LOGE("rule fst: %{public}s", f.c_str());
#else
          SHERPA_ONNX_LOGE("rule fst: %s", f.c_str());
#endif
        }
        auto buf = ReadFile(mgr, f);
        std::istringstream is(std::string(buf.data(), buf.size()));
        tn_list_.push_back(std::make_unique<kaldifst::TextNormalizer>(is));
      }
    }

    if (!config.rule_fars.empty()) {
      std::vector<std::string> files;
      SplitStringToVector(config.rule_fars, ",", false, &files);
      tn_list_.reserve(files.size() + tn_list_.size());

      for (const auto &f : files) {
        if (config.model.debug) {
#if __OHOS__
          SHERPA_ONNX_LOGE("rule far: %{public}s", f.c_str());
#else
          SHERPA_ONNX_LOGE("rule far: %s", f.c_str());
#endif
        }

        auto buf = ReadFile(mgr, f);

        std::unique_ptr<std::istream> s(
            new std::istringstream(std::string(buf.data(), buf.size())));

        std::unique_ptr<fst::FarReader<fst::StdArc>> reader(
            fst::FarReader<fst::StdArc>::Open(std::move(s)));

        for (; !reader->Done(); reader->Next()) {
          std::unique_ptr<fst::StdConstFst> r(
              fst::CastOrConvertToConstFst(reader->GetFst()->Copy()));

          tn_list_.push_back(
              std::make_unique<kaldifst::TextNormalizer>(std::move(r)));
        }  // for (; !reader->Done(); reader->Next())
      }    // for (const auto &f : files)
    }      // if (!config.rule_fars.empty())
  }

  int32_t SampleRate() const override {
    return model_->GetMetaData().sample_rate;
  }

  int32_t NumSpeakers() const override {
    return model_->GetMetaData().num_speakers;
  }

  // Supported options in GenerationConfig:
  //   - sid: Speaker ID for multi-speaker models
  //   - speed: Speech speed factor. If left at 1.0, it falls back to the
  //            default implied by kokoro.length_scale.
  //   - silence_scale: Scale applied to pauses in the generated audio. If left
  //                    at 0.2, it falls back to OfflineTtsConfig.silence_scale.
  //
  // Supported extra options in config.extra:
  //   - lang: Language override for Kokoro >= 1.0. Defaults to
  //           kokoro.lang if provided, otherwise meta_data.voice.
  GeneratedAudio Generate(
      const std::string &_text, const GenerationConfig &gen_config,
      GeneratedAudioCallback callback = nullptr) const override {
    if (config_.model.debug) {
      SHERPA_ONNX_LOGE("%s", gen_config.ToString().c_str());
    }

    int64_t sid = gen_config.sid;
    float speed = gen_config.speed;
    if (speed <= 0) {
      SHERPA_ONNX_LOGE("Speed must be > 0. Given: %f", speed);
      return {};
    }

    const auto &meta_data = model_->GetMetaData();
    int32_t num_speakers = meta_data.num_speakers;

    if (num_speakers == 0 && sid != 0) {
#if __OHOS__
      SHERPA_ONNX_LOGE(
          "This is a single-speaker model and supports only sid 0. Given sid: "
          "%{public}d. sid is ignored",
          static_cast<int32_t>(sid));
#else
      SHERPA_ONNX_LOGE(
          "This is a single-speaker model and supports only sid 0. Given sid: "
          "%d. sid is ignored",
          static_cast<int32_t>(sid));
#endif
    }

    if (num_speakers != 0 && (sid >= num_speakers || sid < 0)) {
#if __OHOS__
      SHERPA_ONNX_LOGE(
          "This model contains only %{public}d speakers. sid should be in the "
          "range [%{public}d, %{public}d]. Given: %{public}d. Use sid=0",
          num_speakers, 0, num_speakers - 1, static_cast<int32_t>(sid));
#else
      SHERPA_ONNX_LOGE(
          "This model contains only %d speakers. sid should be in the range "
          "[%d, %d]. Given: %d. Use sid=0",
          num_speakers, 0, num_speakers - 1, static_cast<int32_t>(sid));
#endif
      sid = 0;
    }

    std::string text = _text;
    if (config_.model.debug) {
#if __OHOS__
      SHERPA_ONNX_LOGE("Raw text: %{public}s", text.c_str());
#else
      SHERPA_ONNX_LOGE("Raw text: %s", text.c_str());
#endif
      std::ostringstream os;
      os << "In bytes (hex):\n";
      const auto p = reinterpret_cast<const uint8_t *>(text.c_str());
      for (int32_t i = 0; i != text.size(); ++i) {
        os << std::setw(2) << std::setfill('0') << std::hex
           << static_cast<uint32_t>(p[i]) << " ";
      }
      os << "\n";

#if __OHOS__
      SHERPA_ONNX_LOGE("%{public}s", os.str().c_str());
#else
      SHERPA_ONNX_LOGE("%s", os.str().c_str());
#endif
    }

    if (!tn_list_.empty()) {
      for (const auto &tn : tn_list_) {
        text = tn->Normalize(text);
        if (config_.model.debug) {
#if __OHOS__
          SHERPA_ONNX_LOGE("After normalizing: %{public}s", text.c_str());
#else
          SHERPA_ONNX_LOGE("After normalizing: %s", text.c_str());
#endif
        }
      }
    }

    std::string lang = gen_config.GetExtraString("lang");
    if (lang.empty()) {
      lang = config_.model.kokoro.lang.empty() ? meta_data.voice
                                               : config_.model.kokoro.lang;
    }

    std::vector<TokenIDs> token_ids = frontend_->ConvertTextToTokenIds(
        text, lang);

    if (token_ids.empty() ||
        (token_ids.size() == 1 && token_ids[0].tokens.empty())) {
#if __OHOS__
      SHERPA_ONNX_LOGE("Failed to convert '%{public}s' to token IDs",
                       text.c_str());
#else
      SHERPA_ONNX_LOGE("Failed to convert '%s' to token IDs", text.c_str());
#endif
      return {};
    }

    std::vector<std::vector<int64_t>> x;

    x.reserve(token_ids.size());

    for (auto &i : token_ids) {
      x.push_back(std::move(i.tokens));
    }

    int32_t x_size = static_cast<int32_t>(x.size());

    if (config_.max_num_sentences != 1) {
#if __OHOS__
      SHERPA_ONNX_LOGE(
          "max_num_sentences (%{public}d) != 1 is ignored for Kokoro TTS "
          "models",
          config_.max_num_sentences);
#else
      SHERPA_ONNX_LOGE(
          "max_num_sentences (%d) != 1 is ignored for Kokoro TTS models",
          config_.max_num_sentences);
#endif
    }

    // the input text is too long, we process sentences within it in batches
    // to avoid OOM. Batch size is config_.max_num_sentences
    std::vector<std::vector<int64_t>> batch_x;

    int32_t batch_size = 1;
    batch_x.reserve(config_.max_num_sentences);
    int32_t num_batches = x_size / batch_size;

    if (config_.model.debug) {
#if __OHOS__
      SHERPA_ONNX_LOGE(
          "Split it into %{public}d batches. batch size: "
          "%{public}d. Number of sentences: %{public}d",
          num_batches, batch_size, x_size);
#else
      SHERPA_ONNX_LOGE(
          "Split it into %d batches. batch size: %d. Number "
          "of sentences: %d",
          num_batches, batch_size, x_size);
#endif
    }

    GeneratedAudio ans;

    int32_t should_continue = 1;

    int32_t k = 0;

    for (int32_t b = 0; b != num_batches && should_continue; ++b) {
      batch_x.clear();
      for (int32_t i = 0; i != batch_size; ++i, ++k) {
        batch_x.push_back(std::move(x[k]));
      }

      auto audio =
          Process(batch_x, sid, speed, gen_config.silence_scale);
      ans.sample_rate = audio.sample_rate;
      ans.samples.insert(ans.samples.end(), audio.samples.begin(),
                         audio.samples.end());
      if (callback) {
        should_continue = callback(audio.samples.data(), audio.samples.size(),
                                   (b + 1) * 1.0 / num_batches);
        // Caution(fangjun): audio is freed when the callback returns, so users
        // should copy the data if they want to access the data after
        // the callback returns to avoid segmentation fault.
      }
    }

    batch_x.clear();
    while (k < static_cast<int32_t>(x.size()) && should_continue) {
      batch_x.push_back(std::move(x[k]));

      ++k;
    }

    if (!batch_x.empty()) {
      auto audio =
          Process(batch_x, sid, speed, gen_config.silence_scale);
      ans.sample_rate = audio.sample_rate;
      ans.samples.insert(ans.samples.end(), audio.samples.begin(),
                         audio.samples.end());
      if (callback) {
        callback(audio.samples.data(), audio.samples.size(), 1.0);
        // Caution(fangjun): audio is freed when the callback returns, so users
        // should copy the data if they want to access the data after
        // the callback returns to avoid segmentation fault.
      }
    }

    return ans;
  }

  [[deprecated("Use Generate(text, GenerationConfig, callback) instead")]]
  GeneratedAudio Generate(
      const std::string &text, int64_t sid = 0, float speed = 1.0,
      GeneratedAudioCallback callback = nullptr) const override {
    GenerationConfig gen_config;
    gen_config.sid = sid;
    gen_config.speed = speed;
    gen_config.silence_scale = config_.silence_scale;
    if (!config_.model.kokoro.lang.empty()) {
      gen_config.extra["lang"] = config_.model.kokoro.lang;
    }

    return Generate(text, gen_config, std::move(callback));
  }

 private:
  template <typename Manager>
  void InitFrontend(Manager *mgr) {
    const auto &meta_data = model_->GetMetaData();

    if (meta_data.version >= 2) {
      // this is a multi-lingual model, we require that you pass lexicon
      if (config_.model.kokoro.lexicon.empty() &&
          config_.model.kokoro.lang.empty()) {
        SHERPA_ONNX_LOGE("Current model version: '%d'", meta_data.version);
        SHERPA_ONNX_LOGE(
            "You are using a multi-lingual Kokoro model (e.g., Kokoro >= "
            "v1.0). Please pass --kokoro-lexicon or provide --kokoro-lang");
        SHERPA_ONNX_EXIT(-1);
      }

      frontend_ = std::make_unique<KokoroMultiLangLexicon>(
          mgr, config_.model.kokoro.tokens, config_.model.kokoro.lexicon,
          config_.model.kokoro.data_dir, meta_data, config_.model.debug);

      return;
    }

    frontend_ = std::make_unique<PiperPhonemizeLexicon>(
        mgr, config_.model.kokoro.tokens, config_.model.kokoro.data_dir,
        meta_data);
  }

  void InitFrontend() {
    const auto &meta_data = model_->GetMetaData();
    if (meta_data.version >= 2) {
      // this is a multi-lingual model, we require that you pass lexicon
      if (config_.model.kokoro.lexicon.empty() &&
          config_.model.kokoro.lang.empty()) {
        SHERPA_ONNX_LOGE("Current model version: '%d'", meta_data.version);
        SHERPA_ONNX_LOGE(
            "You are using a multi-lingual Kokoro model (e.g., Kokoro >= "
            "v1.0). please pass --kokoro-lexicon or --kokoro-lang");
        SHERPA_ONNX_EXIT(-1);
      }

      frontend_ = std::make_unique<KokoroMultiLangLexicon>(
          config_.model.kokoro.tokens, config_.model.kokoro.lexicon,
          config_.model.kokoro.data_dir, meta_data, config_.model.debug);

      return;
    }

    // this is for kokoro v0.19, which supports only English
    frontend_ = std::make_unique<PiperPhonemizeLexicon>(
        config_.model.kokoro.tokens, config_.model.kokoro.data_dir, meta_data);
  }

  GeneratedAudio Process(const std::vector<std::vector<int64_t>> &tokens,
                         int32_t sid, float speed,
                         float silence_scale) const {
    int32_t num_tokens = 0;
    for (const auto &k : tokens) {
      num_tokens += k.size();
    }

    std::vector<int64_t> x;
    x.reserve(num_tokens);
    for (const auto &k : tokens) {
      x.insert(x.end(), k.begin(), k.end());
    }

    auto memory_info =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

    std::array<int64_t, 2> x_shape = {1, static_cast<int32_t>(x.size())};
    Ort::Value x_tensor = Ort::Value::CreateTensor(
        memory_info, x.data(), x.size(), x_shape.data(), x_shape.size());

    Ort::Value audio = model_->Run(std::move(x_tensor), sid, speed);

    std::vector<int64_t> audio_shape =
        audio.GetTensorTypeAndShapeInfo().GetShape();

    int64_t total = 1;
    // The output shape may be (1, 1, total) or (1, total) or (total,)
    for (auto i : audio_shape) {
      total *= i;
    }

    const float *p = audio.GetTensorData<float>();

    GeneratedAudio ans;
    ans.sample_rate = model_->GetMetaData().sample_rate;
    ans.samples = std::vector<float>(p, p + total);

    if (silence_scale == 0.2f) {
      silence_scale = config_.silence_scale;
    }

    if (silence_scale != 1) {
      ans = ans.ScaleSilence(silence_scale);
    }

    return ans;
  }

 private:
  OfflineTtsConfig config_;
  std::unique_ptr<OfflineTtsKokoroModel> model_;
  std::vector<std::unique_ptr<kaldifst::TextNormalizer>> tn_list_;
  std::unique_ptr<OfflineTtsFrontend> frontend_;
};

}  // namespace sherpa_onnx
#endif  // SHERPA_ONNX_CSRC_OFFLINE_TTS_KOKORO_IMPL_H_
