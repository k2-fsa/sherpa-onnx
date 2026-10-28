// sherpa-onnx/csrc/offline-tts-vits-impl.h
//
// Copyright (c)  2023-2026  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_OFFLINE_TTS_VITS_IMPL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_TTS_VITS_IMPL_H_

#include <algorithm>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/character-lexicon.h"
#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/lexicon.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/melo-tts-lexicon.h"
#include "sherpa-onnx/csrc/offline-tts-character-frontend.h"
#include "sherpa-onnx/csrc/offline-tts-frontend.h"
#include "sherpa-onnx/csrc/offline-tts-impl.h"
#include "sherpa-onnx/csrc/offline-tts-vits-model.h"
#include "sherpa-onnx/csrc/text-utils.h"
#include "sherpa-onnx/csrc/tts-text-normalizer.h"

namespace sherpa_onnx {

class OfflineTtsVitsImpl : public OfflineTtsImpl {
 public:
  explicit OfflineTtsVitsImpl(const OfflineTtsConfig &config)
      : config_(config),
        model_(std::make_unique<OfflineTtsVitsModel>(config.model)) {
    InitFrontend();

    if (!frontend_ && !config.model.vits.tokens.empty()) {
      auto is = OpenInputFile(config.model.vits.tokens);
      token2id_ = ReadPiperTokens(is);
    }

    if (!frontend_ && !config.model.vits.lexicon.empty()) {
      auto is = OpenInputFile(config.model.vits.lexicon);
      LoadLexicon(is, config.model.debug);
    }

    tn_list_ = LoadTextNormalizers(config.rule_fsts, config.rule_fars,
                                   config.model.debug);
  }

  template <typename Manager>
  OfflineTtsVitsImpl(Manager *mgr, const OfflineTtsConfig &config)
      : config_(config),
        model_(std::make_unique<OfflineTtsVitsModel>(mgr, config.model)) {
    InitFrontend(mgr);

    if (!frontend_ && !config.model.vits.tokens.empty()) {
      auto buf = ReadFile(mgr, config.model.vits.tokens);
      std::istringstream is(std::string(buf.data(), buf.size()));
      token2id_ = ReadPiperTokens(is);
    }

    if (!frontend_ && !config.model.vits.lexicon.empty()) {
      auto buf = ReadFile(mgr, config.model.vits.lexicon);
      std::istringstream is(std::string(buf.data(), buf.size()));
      LoadLexicon(is, config.model.debug);
    }

    tn_list_ = LoadTextNormalizers(mgr, config.rule_fsts, config.rule_fars,
                                   config.model.debug);
  }

  int32_t SampleRate() const override {
    return model_->GetMetaData().sample_rate;
  }

  int32_t NumSpeakers() const override {
    return model_->GetMetaData().num_speakers;
  }

  std::string Lang() const override { return LastLang(model_->GetMetaData().voice); }

  GeneratedAudio Generate(
      const std::string &_text, const GenerationConfig &gen_config,
      GeneratedAudioCallback callback = nullptr) const override {
    bool debug = gen_config.GetExtraInt("debug", config_.model.debug);

    if (debug) {
      SHERPA_ONNX_LOGE("%s", gen_config.ToString().c_str());
    }

    const auto &meta_data = model_->GetMetaData();

    float speed = gen_config.speed;
    if (speed <= 0) {
      SHERPA_ONNX_LOGE("Speed must be > 0. Given: %f", speed);
      return {};
    }

    int64_t sid = ValidateSid(gen_config.sid, meta_data.num_speakers);
    int64_t emotion_id = ValidateEmotionId(
        gen_config.GetExtraInt("emotion_id", 0), meta_data.num_emotions);

    std::string text = _text;
    if (!text.empty()) {
      text = NormalizeText(text, debug);
    }

    std::vector<TokenIDs> token_ids =
        Tokenize(gen_config, text, meta_data, debug);
    if (token_ids.empty() ||
        (token_ids.size() == 1 && token_ids[0].tokens.empty())) {
      SHERPA_ONNX_LOGE("Failed to convert '%s' to token IDs", text.c_str());
      return {};
    }

    std::vector<std::vector<int64_t>> x;
    std::vector<std::vector<int64_t>> tones;

    x.reserve(token_ids.size());
    for (auto &i : token_ids) {
      x.push_back(std::move(i.tokens));
    }

    if (!token_ids[0].tones.empty()) {
      tones.reserve(token_ids.size());
      for (auto &i : token_ids) {
        tones.push_back(std::move(i.tones));
      }
    }

    if (gen_config.phoneme_codepoints.empty() && meta_data.add_blank &&
        meta_data.frontend != "characters") {
      for (auto &k : x) {
        k = AddBlank(k);
      }
      for (auto &k : tones) {
        k = AddBlank(k);
      }
    }

    if (debug) {
      for (int32_t i = 0; i < static_cast<int32_t>(x.size()); ++i) {
        std::ostringstream os;
        os << "sentence " << i << ": " << x[i].size() << " token IDs: [";
        for (int32_t j = 0; j < static_cast<int32_t>(x[i].size()); ++j) {
          if (j > 0) os << ", ";
          os << x[i][j];
        }
        os << "]";
#if __OHOS__
        SHERPA_ONNX_LOGE("%{public}s", os.str().c_str());
#else
        SHERPA_ONNX_LOGE("%s", os.str().c_str());
#endif
      }
    }

    return GenerateInBatches(x, tones, sid, speed, gen_config.silence_scale,
                             emotion_id, debug, callback);
  }

  [[deprecated("Use Generate(text, GenerationConfig, callback) instead")]]
  GeneratedAudio Generate(
      const std::string &text, int64_t sid = 0, float speed = 1.0,
      GeneratedAudioCallback callback = nullptr) const override {
    GenerationConfig gen_config;
    gen_config.sid = sid;
    gen_config.speed = speed;
    gen_config.silence_scale = config_.silence_scale;
    return Generate(text, gen_config, std::move(callback));
  }

 private:
  int64_t ValidateSid(int64_t sid, int32_t num_speakers) const {
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

    return sid;
  }

  int64_t ValidateEmotionId(int64_t emotion_id, int32_t num_emotions) const {
    if (num_emotions == 0 && emotion_id != 0) {
#if __OHOS__
      SHERPA_ONNX_LOGE(
          "This model does not support emotion selection. Given emotion_id: "
          "%{public}d. emotion_id is ignored",
          static_cast<int32_t>(emotion_id));
#else
      SHERPA_ONNX_LOGE(
          "This model does not support emotion selection. Given emotion_id: "
          "%d. emotion_id is ignored",
          static_cast<int32_t>(emotion_id));
#endif
      emotion_id = 0;
    }

    if (num_emotions != 0 && (emotion_id >= num_emotions || emotion_id < 0)) {
#if __OHOS__
      SHERPA_ONNX_LOGE(
          "This model contains only %{public}d emotions. emotion_id should be "
          "in the range [%{public}d, %{public}d]. Given: %{public}d. Use "
          "emotion_id=0",
          num_emotions, 0, num_emotions - 1, static_cast<int32_t>(emotion_id));
#else
      SHERPA_ONNX_LOGE(
          "This model contains only %d emotions. emotion_id should be in the "
          "range [%d, %d]. Given: %d. Use emotion_id=0",
          num_emotions, 0, num_emotions - 1, static_cast<int32_t>(emotion_id));
#endif
      emotion_id = 0;
    }

    return emotion_id;
  }

  std::string NormalizeText(const std::string &text, bool debug) const {
    // Canonical composition first (NFC), so that e.g. "a" + U+0301 written in
    // the input matches a lexicon entry stored as U+00E1. Must run before any
    // lowercasing.
    std::string result = NormalizeNfc(text);

    if (debug) {
#if __OHOS__
      SHERPA_ONNX_LOGE("Raw text: %{public}s", result.c_str());
#else
      SHERPA_ONNX_LOGE("Raw text: %s", result.c_str());
#endif
    }

    if (!tn_list_.empty()) {
      for (const auto &tn : tn_list_) {
        result = tn->Normalize(result);
        if (debug) {
#if __OHOS__
          SHERPA_ONNX_LOGE("After normalizing: %{public}s", result.c_str());
#else
          SHERPA_ONNX_LOGE("After normalizing: %s", result.c_str());
#endif
        }
      }
    }

    return result;
  }

  std::vector<TokenIDs> Tokenize(const GenerationConfig &gen_config,
                                 const std::string &text,
                                 const OfflineTtsVitsModelMetaData &meta_data,
                                 bool debug) const {
    if (!gen_config.tokens.empty()) {
      // Pre-tokenized input given as token strings. Each inner vector is one
      // sentence. VITS vocabularies key one symbol per Unicode codepoint (see
      // ReadPiperTokens), so convert to codepoints and reuse the codepoint
      // path: that way the bos/pad/eos markers are added here, exactly as for
      // phoneme_codepoints, and callers pass bare tokens.
      std::vector<std::vector<int32_t>> codepoints;
      codepoints.reserve(gen_config.tokens.size());
      for (const auto &sentence : gen_config.tokens) {
        std::vector<int32_t> cps;
        cps.reserve(sentence.size());
        for (const auto &tok : sentence) {
          const std::u32string u32 = Utf8ToUtf32(tok);
          if (u32.empty()) {
            continue;
          }
          if (u32.size() != 1) {
            SHERPA_ONNX_LOGE(
                "Token '%s' in GenerationConfig.tokens is not a single "
                "codepoint and cannot be used with this model. It is ignored.",
                tok.c_str());
            continue;
          }
          cps.push_back(static_cast<int32_t>(u32[0]));
        }
        if (!cps.empty()) {
          codepoints.emplace_back(std::move(cps));
        }
      }
      return TokenizeFromCodepoints(codepoints);
    }

    if (!gen_config.phoneme_codepoints.empty()) {
      return TokenizeFromCodepoints(gen_config.phoneme_codepoints);
    }

    if (!lexicon_.empty()) {
      auto codepoints = TokenizeFromLexicon(text, debug);
      return TokenizeFromCodepoints(codepoints);
    }

    if (frontend_) {
      return frontend_->ConvertTextToTokenIds(text, meta_data.voice);
    }

    SHERPA_ONNX_LOGE(
        "phoneme_codepoints is empty, no lexicon, and no frontend available. "
        "Please provide phoneme_codepoints via GenerationConfig or "
        "a lexicon via --vits-lexicon.");
    return {};
  }

  GeneratedAudio GenerateInBatches(std::vector<std::vector<int64_t>> &x,
                                   std::vector<std::vector<int64_t>> &tones,
                                   int64_t sid, float speed,
                                   float silence_scale, int64_t emotion_id,
                                   bool debug,
                                   GeneratedAudioCallback callback) const {
    int32_t x_size = static_cast<int32_t>(x.size());

    if (config_.max_num_sentences <= 0 || x_size <= config_.max_num_sentences) {
      auto ans = Process(x, tones, sid, speed, silence_scale, emotion_id);
      if (callback) {
        callback(ans.samples.data(), ans.samples.size(), 1.0);
      }
      return ans;
    }

    std::vector<std::vector<int64_t>> batch_x;
    std::vector<std::vector<int64_t>> batch_tones;

    int32_t batch_size = config_.max_num_sentences;
    batch_x.reserve(batch_size);
    batch_tones.reserve(batch_size);
    int32_t num_batches = x_size / batch_size;

    if (debug) {
#if __OHOS__
      SHERPA_ONNX_LOGE(
          "Text is too long. Split it into %{public}d batches. batch size: "
          "%{public}d. Number of sentences: %{public}d",
          num_batches, batch_size, x_size);
#else
      SHERPA_ONNX_LOGE(
          "Text is too long. Split it into %d batches. batch size: %d. Number "
          "of sentences: %d",
          num_batches, batch_size, x_size);
#endif
    }

    GeneratedAudio ans;
    int32_t should_continue = 1;
    int32_t k = 0;

    for (int32_t b = 0; b != num_batches && should_continue; ++b) {
      batch_x.clear();
      batch_tones.clear();
      for (int32_t i = 0; i != batch_size; ++i, ++k) {
        batch_x.push_back(std::move(x[k]));
        if (!tones.empty()) {
          batch_tones.push_back(std::move(tones[k]));
        }
      }

      auto audio =
          Process(batch_x, batch_tones, sid, speed, silence_scale, emotion_id);
      ans.sample_rate = audio.sample_rate;
      ans.samples.insert(ans.samples.end(), audio.samples.begin(),
                         audio.samples.end());
      if (callback) {
        should_continue = callback(audio.samples.data(), audio.samples.size(),
                                   (b + 1) * 1.0 / num_batches);
      }
    }

    batch_x.clear();
    batch_tones.clear();
    while (k < x_size && should_continue) {
      batch_x.push_back(std::move(x[k]));
      if (!tones.empty()) {
        batch_tones.push_back(std::move(tones[k]));
      }
      ++k;
    }

    if (!batch_x.empty()) {
      auto audio =
          Process(batch_x, batch_tones, sid, speed, silence_scale, emotion_id);
      ans.sample_rate = audio.sample_rate;
      ans.samples.insert(ans.samples.end(), audio.samples.begin(),
                         audio.samples.end());
      if (callback) {
        callback(audio.samples.data(), audio.samples.size(), 1.0);
      }
    }

    return ans;
  }

  std::vector<TokenIDs> TokenizeFromCodepoints(
      const std::vector<std::vector<int32_t>> &codepoints) const {
    const auto &meta_data = model_->GetMetaData();
    std::vector<TokenIDs> result;
    result.reserve(codepoints.size());

    for (const auto &phonemes : codepoints) {
      std::vector<char32_t> phonemes_c32(phonemes.begin(), phonemes.end());

      if (meta_data.is_piper || meta_data.is_icefall || meta_data.is_inflect) {
        auto ids = PiperPhonemesToIdsVits(token2id_, phonemes_c32,
                                          meta_data.is_inflect);
        result.emplace_back(std::move(ids));
      } else if (meta_data.is_coqui) {
        auto ids = CoquiPhonemesToIds(token2id_, phonemes_c32, meta_data);
        result.emplace_back(std::move(ids));
      } else {
        SHERPA_ONNX_LOGE(
            "phoneme_codepoints not supported for this model type");
        return {};
      }
    }
    return result;
  }

  static bool IsNewFormatLexicon(const std::string &path) {
    auto is = OpenInputFile(path);
    return IsNewFormatLexiconStream(is);
  }

  // Android/OHOS assets cannot be opened with ifstream, so read the file
  // with the asset-aware reader before checking its format. Without this,
  // a new-format lexicon shipped as an asset is misdetected as old-format
  // and every entry is dropped.
  template <typename Manager>
  static bool IsNewFormatLexicon(Manager *mgr, const std::string &path) {
    auto buf = ReadFile(mgr, path);
    std::istringstream is(std::string(buf.data(), buf.size()));
    return IsNewFormatLexiconStream(is);
  }

  static bool IsNewFormatLexiconStream(std::istream &is) {
    std::string line;
    while (std::getline(is, line)) {
      if (line.empty() || line[0] == '#') continue;
      return line.find("||") != std::string::npos;
    }
    return false;
  }

  void InitFrontend() {
    const auto &meta_data = model_->GetMetaData();

    if (meta_data.frontend == "characters") {
      frontend_ = std::make_unique<OfflineTtsCharacterFrontend>(
          config_.model.vits.tokens, meta_data);
    } else if (meta_data.jieba && meta_data.is_melo_tts) {
      frontend_ = std::make_unique<MeloTtsLexicon>(
          config_.model.vits.lexicon, config_.model.vits.tokens,
          model_->GetMetaData(), config_.model.debug);
    } else if (meta_data.is_melo_tts && meta_data.language == "English") {
      frontend_ = std::make_unique<MeloTtsLexicon>(
          config_.model.vits.lexicon, config_.model.vits.tokens,
          model_->GetMetaData(), config_.model.debug);
    } else if (meta_data.jieba || meta_data.use_g2pw) {
      frontend_ = std::make_unique<CharacterLexicon>(
          config_.model.vits.lexicon, config_.model.vits.tokens,
          config_.model.debug, meta_data.use_g2pw);
    } else if (meta_data.is_piper || meta_data.is_coqui ||
               meta_data.is_icefall || meta_data.is_inflect) {
      // No frontend — uses phoneme_codepoints or new-format lexicon
      // If an old-format lexicon is provided (no ||), fall through to Lexicon
      if (!config_.model.vits.lexicon.empty() &&
          !IsNewFormatLexicon(config_.model.vits.lexicon)) {
        frontend_ = std::make_unique<Lexicon>(
            config_.model.vits.lexicon, config_.model.vits.tokens,
            meta_data.punctuations, meta_data.language, config_.model.debug);
      }
    } else if (!config_.model.vits.lexicon.empty()) {
      frontend_ = std::make_unique<Lexicon>(
          config_.model.vits.lexicon, config_.model.vits.tokens,
          meta_data.punctuations, meta_data.language, config_.model.debug);
    } else {
      SHERPA_ONNX_LOGE(
          "No frontend available. For piper/icefall/inflect/coqui models, "
          "provide phoneme_codepoints via GenerationConfig at generate time.");
    }
  }

  template <typename Manager>
  void InitFrontend(Manager *mgr) {
    const auto &meta_data = model_->GetMetaData();

    if (meta_data.frontend == "characters") {
      frontend_ = std::make_unique<OfflineTtsCharacterFrontend>(
          mgr, config_.model.vits.tokens, meta_data);
    } else if (meta_data.jieba && meta_data.is_melo_tts) {
      frontend_ = std::make_unique<MeloTtsLexicon>(
          mgr, config_.model.vits.lexicon, config_.model.vits.tokens,
          model_->GetMetaData(), config_.model.debug);
    } else if (meta_data.jieba || meta_data.use_g2pw) {
      frontend_ = std::make_unique<CharacterLexicon>(
          mgr, config_.model.vits.lexicon, config_.model.vits.tokens,
          config_.model.debug, meta_data.use_g2pw);
    } else if (meta_data.is_melo_tts && meta_data.language == "English") {
      frontend_ = std::make_unique<MeloTtsLexicon>(
          mgr, config_.model.vits.lexicon, config_.model.vits.tokens,
          model_->GetMetaData(), config_.model.debug);
    } else if (meta_data.is_piper || meta_data.is_coqui ||
               meta_data.is_icefall || meta_data.is_inflect) {
      if (!config_.model.vits.lexicon.empty() &&
          !IsNewFormatLexicon(mgr, config_.model.vits.lexicon)) {
        frontend_ = std::make_unique<Lexicon>(
            mgr, config_.model.vits.lexicon, config_.model.vits.tokens,
            meta_data.punctuations, meta_data.language, config_.model.debug);
      }
    } else if (!config_.model.vits.lexicon.empty()) {
      frontend_ = std::make_unique<Lexicon>(
          mgr, config_.model.vits.lexicon, config_.model.vits.tokens,
          meta_data.punctuations, meta_data.language, config_.model.debug);
    } else {
      SHERPA_ONNX_LOGE(
          "No frontend available. For piper/icefall/inflect/coqui models, "
          "provide phoneme_codepoints via GenerationConfig at generate time.");
    }
  }

  GeneratedAudio Process(const std::vector<std::vector<int64_t>> &tokens,
                         const std::vector<std::vector<int64_t>> &tones,
                         int32_t sid, float speed, float silence_scale,
                         int64_t emotion_id = 0) const {
    int32_t num_tokens = 0;
    for (const auto &k : tokens) {
      num_tokens += k.size();
    }

    std::vector<int64_t> x;
    x.reserve(num_tokens);
    for (const auto &k : tokens) {
      x.insert(x.end(), k.begin(), k.end());
    }

    std::vector<int64_t> tone_list;
    if (!tones.empty()) {
      tone_list.reserve(num_tokens);
      for (const auto &k : tones) {
        tone_list.insert(tone_list.end(), k.begin(), k.end());
      }
    }

    auto memory_info =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

    std::array<int64_t, 2> x_shape = {1, static_cast<int32_t>(x.size())};
    Ort::Value x_tensor = Ort::Value::CreateTensor(
        memory_info, x.data(), x.size(), x_shape.data(), x_shape.size());

    Ort::Value tones_tensor{nullptr};
    if (!tones.empty()) {
      tones_tensor = Ort::Value::CreateTensor(memory_info, tone_list.data(),
                                              tone_list.size(), x_shape.data(),
                                              x_shape.size());
    }

    Ort::Value audio{nullptr};
    if (tones.empty()) {
      audio = model_->Run(std::move(x_tensor), sid, speed, emotion_id);
    } else {
      audio =
          model_->Run(std::move(x_tensor), std::move(tones_tensor), sid, speed);
    }

    std::vector<int64_t> audio_shape =
        audio.GetTensorTypeAndShapeInfo().GetShape();

    int64_t total = 1;
    for (auto i : audio_shape) {
      total *= i;
    }

    const float *p = audio.GetTensorData<float>();

    GeneratedAudio ans;
    ans.sample_rate = model_->GetMetaData().sample_rate;
    ans.samples = std::vector<float>(p, p + total);

    if (silence_scale != 1) {
      ans = ans.ScaleSilence(silence_scale);
    }

    return ans;
  }

  void LoadLexicon(std::istream &is, bool debug) {
    auto entries = ParseLexiconFile(is, &max_lexicon_phrase_len_);
    for (auto &e : entries) {
      std::vector<int32_t> codepoints;
      for (const auto &p : e.phonemes) {
        std::u32string u32 = Utf8ToUtf32(p);
        for (char32_t cp : u32) {
          codepoints.push_back(static_cast<int32_t>(cp));
        }
      }
      lexicon_[e.key] = std::move(codepoints);
    }
    if (debug) {
      SHERPA_ONNX_LOGE("Loaded lexicon with %d entries, max phrase len %d",
                       static_cast<int32_t>(lexicon_.size()),
                       max_lexicon_phrase_len_);
    }
  }

  std::vector<std::vector<int32_t>> TokenizeFromLexicon(const std::string &text,
                                                        bool debug) const {
    std::vector<std::vector<int32_t>> result;

    // split text into sentences by . ! ? ; :
    std::vector<std::string> sentences;
    std::string current;
    for (char c : text) {
      current.push_back(c);
      if (c == '.' || c == '!' || c == '?' || c == ';' || c == ':') {
        sentences.push_back(current);
        current.clear();
      }
    }
    if (!current.empty()) {
      sentences.push_back(current);
    }

    for (const auto &sentence : sentences) {
      // tokenize into words (split by space/tab, keep punctuation separate)
      std::vector<std::string> words;
      std::string word;
      for (char c : sentence) {
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
          if (!word.empty()) {
            words.push_back(word);
            word.clear();
          }
        } else if (c == ',' || c == ';' || c == ':' || c == '.' || c == '!' ||
                   c == '?') {
          if (!word.empty()) {
            words.push_back(word);
            word.clear();
          }
          words.push_back(std::string(1, c));
        } else {
          word.push_back(c);
        }
      }
      if (!word.empty()) {
        words.push_back(word);
      }

      if (words.empty()) {
        continue;
      }

      std::vector<int32_t> sentence_codepoints;
      int32_t i = 0;
      int32_t n = static_cast<int32_t>(words.size());

      while (i < n) {
        // punctuation: pass through as literal codepoints
        if (words[i].size() == 1) {
          char c = words[i][0];
          if (c == ',' || c == '.' || c == '!' || c == '?' || c == ';' ||
              c == ':') {
            // The word boundary that espeak/piper emit after punctuation is
            // added by AppendWordBoundary() before the *next* word, so that a
            // sentence ending in punctuation does not grow a trailing space.
            sentence_codepoints.push_back(static_cast<int32_t>(c));
            ++i;
            continue;
          }
        }

        bool found = false;
        // longest match: try max_lexicon_phrase_len_ words, then fewer
        int32_t max_try = std::min(max_lexicon_phrase_len_, n - i);
        for (int32_t len = max_try; len >= 1; --len) {
          std::string phrase;
          for (int32_t j = i; j < i + len; ++j) {
            if (j > i) phrase.push_back(' ');
            phrase += NormalizeLexiconKey(words[j]);
          }

          auto it = lexicon_.find(phrase);
          if (it != lexicon_.end()) {
            if (debug) {
              std::string phonemes_str;
              for (int32_t cp : it->second) {
                phonemes_str += Utf32ToUtf8(static_cast<char32_t>(cp));
              }
              SHERPA_ONNX_LOGE("Lexicon matched: '%s' -> '%s'", phrase.c_str(),
                               phonemes_str.c_str());
            }
            // Word boundary before the next word (like espeak does). Idempotent
            // w.r.t. the space already emitted after punctuation.
            if (!sentence_codepoints.empty() &&
                sentence_codepoints.back() != static_cast<int32_t>(' ')) {
              sentence_codepoints.push_back(static_cast<int32_t>(' '));
            }
            sentence_codepoints.insert(sentence_codepoints.end(),
                                       it->second.begin(), it->second.end());
            i += len;
            found = true;
            break;
          }
        }

        if (!found) {
          // The whole token already missed in the loop above. Strip/split
          // punctuation and retry -- see LookupPieces().
          LookupPieces(words[i], &sentence_codepoints, debug);
          ++i;
        }
      }

      if (!sentence_codepoints.empty()) {
        if (debug) {
          std::string phonemes_str;
          for (int32_t cp : sentence_codepoints) {
            phonemes_str += Utf32ToUtf8(static_cast<char32_t>(cp));
          }
          SHERPA_ONNX_LOGE(
              "Sentence %d: text='%s', %d codepoints, phonemes='%s'",
              static_cast<int32_t>(result.size()), Trim(sentence).c_str(),
              static_cast<int32_t>(sentence_codepoints.size()),
              phonemes_str.c_str());
        }
        result.push_back(std::move(sentence_codepoints));
      }
    }

    return result;
  }

  // Insert a word boundary unless the stream is empty or already ends with
  // one. Idempotent w.r.t. the space emitted after punctuation.
  static void AppendWordBoundary(std::vector<int32_t> *cps) {
    if (!cps->empty() && cps->back() != static_cast<int32_t>(' ')) {
      cps->push_back(static_cast<int32_t>(' '));
    }
  }

  void LogMatched(const std::string &word, const std::vector<int32_t> &phonemes,
                  bool debug) const {
    if (!debug) return;
    std::string phonemes_str;
    for (int32_t cp : phonemes) {
      phonemes_str += Utf32ToUtf8(static_cast<char32_t>(cp));
    }
    SHERPA_ONNX_LOGE("Lexicon matched: '%s' -> '%s'", word.c_str(),
                     phonemes_str.c_str());
  }

  // Emit phonemes for |word| when the whole token is not in the lexicon.
  //
  // (#4020) Words with punctuation attached -- "Dveme, word—, ‘word’, (word
  // -- form a single token, miss the lexicon and get dropped. Rather than
  // making those characters token separators (which would break tokens the
  // lexicon does contain, e.g. don't), retry the lookup with the punctuation
  // stripped away, and only then split.
  void LookupPieces(const std::string &word,
                    std::vector<int32_t> *sentence_codepoints,
                    bool debug) const {
    std::u32string u32 = Utf8ToUtf32(word);

    // 1. strip leading/trailing non-word characters
    std::size_t b = 0;
    std::size_t e = u32.size();
    while (b < e && !IsLexiconWordChar(u32[b])) ++b;
    while (e > b && !IsLexiconWordChar(u32[e - 1])) --e;

    std::u32string core = u32.substr(b, e - b);
    if (core.empty()) {
      // nothing but punctuation
      SHERPA_ONNX_LOGE("OOV word skipped: '%s'", word.c_str());
      return;
    }

    // 2. retry with the edges stripped
    std::string core_s = Utf32ToUtf8(core);
    auto it = lexicon_.find(NormalizeLexiconKey(core_s));
    if (it != lexicon_.end()) {
      AppendWordBoundary(sentence_codepoints);
      LogMatched(core_s, it->second, debug);
      sentence_codepoints->insert(sentence_codepoints->end(),
                                  it->second.begin(), it->second.end());
      return;
    }

    // 3. still OOV: split on non-word characters and look up each piece. A
    //    boundary owed by a skipped piece survives to the next emitted one, so
    //    dropping a word cannot glue its neighbours together.
    auto pieces = SplitLexiconPieces(core);
    std::vector<bool> needs_boundary;
    std::vector<bool> found;
    std::vector<const std::vector<int32_t> *> matched;
    needs_boundary.reserve(pieces.size());
    found.reserve(pieces.size());
    matched.reserve(pieces.size());

    for (const auto &piece : pieces) {
      needs_boundary.push_back(piece.needs_boundary);
      auto it2 = lexicon_.find(NormalizeLexiconKey(piece.text));
      if (it2 == lexicon_.end()) {
        // Always report dropped words; never hide them behind the debug flag.
        SHERPA_ONNX_LOGE("OOV word skipped: '%s'", piece.text.c_str());
        found.push_back(false);
        matched.push_back(nullptr);
      } else {
        found.push_back(true);
        matched.push_back(&it2->second);
      }
    }

    auto boundaries = AssignLexiconPieceBoundaries(needs_boundary, found);
    for (std::size_t i = 0; i < pieces.size(); ++i) {
      if (!found[i]) continue;
      if (boundaries[i]) {
        AppendWordBoundary(sentence_codepoints);
      }
      LogMatched(pieces[i].text, *matched[i], debug);
      sentence_codepoints->insert(sentence_codepoints->end(),
                                  matched[i]->begin(), matched[i]->end());
    }
  }

 private:
  OfflineTtsConfig config_;
  std::unique_ptr<OfflineTtsVitsModel> model_;
  std::vector<std::unique_ptr<kaldifst::TextNormalizer>> tn_list_;
  std::unordered_map<char32_t, int32_t> token2id_;
  std::unique_ptr<OfflineTtsFrontend> frontend_;
  std::unordered_map<std::string, std::vector<int32_t>> lexicon_;
  int32_t max_lexicon_phrase_len_ = 1;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_TTS_VITS_IMPL_H_
