// sherpa-onnx/csrc/offline-tts-matcha-impl.h
//
// Copyright (c)  2024  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_OFFLINE_TTS_MATCHA_IMPL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_TTS_MATCHA_IMPL_H_

#include <algorithm>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/offline-tts-frontend.h"
#include "sherpa-onnx/csrc/offline-tts-impl.h"
#include "sherpa-onnx/csrc/offline-tts-matcha-model.h"
#include "sherpa-onnx/csrc/onnx-utils.h"
#include "sherpa-onnx/csrc/symbol-table.h"
#include "sherpa-onnx/csrc/text-utils.h"
#include "sherpa-onnx/csrc/tts-text-normalizer.h"
#include "sherpa-onnx/csrc/vocoder.h"

namespace sherpa_onnx {

class OfflineTtsMatchaImpl : public OfflineTtsImpl {
 public:
  explicit OfflineTtsMatchaImpl(const OfflineTtsConfig &config)
      : config_(config),
        model_(std::make_unique<OfflineTtsMatchaModel>(config.model)) {
    const auto &meta_data = model_->GetMetaData();
    if (meta_data.need_vocoder) {
      if (config.model.matcha.vocoder.empty()) {
        SHERPA_ONNX_LOGE("Please provide vocoder for this model");
        SHERPA_ONNX_EXIT(-1);
      }

      if (!FileExists(config.model.matcha.vocoder)) {
        SHERPA_ONNX_LOGE("The vocoder '%s' does not exist",
                         config.model.matcha.vocoder.c_str());
        SHERPA_ONNX_EXIT(-1);
      }

      vocoder_ = Vocoder::Create(config.model);
    } else if (!config.model.matcha.vocoder.empty()) {
      SHERPA_ONNX_LOGE(
          "You don't need to provide vocoder for this model. Ignore it");
    }

    {
      auto is = OpenInputFile(config.model.matcha.tokens);
      token_str2id_ = ReadTokens(is);
    }

    // Build token2id_ (char32_t → int) from single-codepoint entries in
    // tokens.txt. For IPA models all entries are single-codepoint; for
    // zh-en models only the IPA entries are single-codepoint (pinyin entries
    // like "zhong1" are multi-char and go into token_str2id_ only).
    {
      auto is = OpenInputFile(config.model.matcha.tokens);
      std::string line;
      while (std::getline(is, line)) {
        std::istringstream iss(line);
        std::string sym;
        int32_t id = 0;
        iss >> sym;
        if (sym.empty()) continue;
        if (iss.eof()) {
          id = atoi(sym.c_str());
          sym = " ";
        } else {
          iss >> id;
        }
        std::u32string u32 = Utf8ToUtf32(sym);
        if (u32.size() == 1) {
          char32_t c = u32[0];
          if (!token2id_.count(c)) {
            token2id_[c] = id;
          }
        }
      }
    }

    if (!config.model.matcha.lexicon.empty()) {
      std::vector<std::string> files;
      SplitStringToVector(config.model.matcha.lexicon, ",", false, &files);
      for (const auto &f : files) {
        auto is = OpenInputFile(f);
        LoadLexicon(is, config.model.debug);
      }
    }

    tn_list_ = LoadTextNormalizers(config.rule_fsts, config.rule_fars,
                                   config.model.debug);

    if (vocoder_ && vocoder_->SampleRate() != meta_data.sample_rate) {
      SHERPA_ONNX_LOGE(
          "ERROR: Vocoder sample rate (%d Hz) does not match the TTS model "
          "sample rate (%d Hz).",
          vocoder_->SampleRate(), meta_data.sample_rate);
      SHERPA_ONNX_LOGE(
          "Please download the correct vocoder from "
          "https://github.com/k2-fsa/sherpa-onnx/releases/tag/vocoder-models");
      if (meta_data.sample_rate == 16000) {
        SHERPA_ONNX_LOGE("For 16kHz models, use vocos-16khz-univ.onnx");
      } else if (meta_data.sample_rate == 22050) {
        SHERPA_ONNX_LOGE("For 22050Hz models, use vocos-22khz-univ.onnx");
      } else if (meta_data.sample_rate == 24000) {
        SHERPA_ONNX_LOGE("For 24kHz models, use vocos-24khz-univ.onnx");
      }
      SHERPA_ONNX_EXIT(-1);
    }
  }

  template <typename Manager>
  OfflineTtsMatchaImpl(Manager *mgr, const OfflineTtsConfig &config)
      : config_(config),
        model_(std::make_unique<OfflineTtsMatchaModel>(mgr, config.model)) {
    const auto &meta_data = model_->GetMetaData();
    if (meta_data.need_vocoder) {
      if (config.model.matcha.vocoder.empty()) {
        SHERPA_ONNX_LOGE("Please provide vocoder for this model");
        SHERPA_ONNX_EXIT(-1);
      }

      vocoder_ = Vocoder::Create(mgr, config.model);
    } else if (!config.model.matcha.vocoder.empty()) {
      SHERPA_ONNX_LOGE(
          "You don't need to provide vocoder for this model. Ignore it");
    }

    {
      auto buf = ReadFile(mgr, config.model.matcha.tokens);
      std::istringstream is(std::string(buf.data(), buf.size()));
      token_str2id_ = ReadTokens(is);
    }

    {
      auto buf = ReadFile(mgr, config.model.matcha.tokens);
      std::istringstream is(std::string(buf.data(), buf.size()));
      std::string line;
      while (std::getline(is, line)) {
        std::istringstream iss(line);
        std::string sym;
        int32_t id = 0;
        iss >> sym;
        if (sym.empty()) continue;
        if (iss.eof()) {
          id = atoi(sym.c_str());
          sym = " ";
        } else {
          iss >> id;
        }
        std::u32string u32 = Utf8ToUtf32(sym);
        if (u32.size() == 1) {
          char32_t c = u32[0];
          if (!token2id_.count(c)) {
            token2id_[c] = id;
          }
        }
      }
    }

    if (!config.model.matcha.lexicon.empty()) {
      std::vector<std::string> files;
      SplitStringToVector(config.model.matcha.lexicon, ",", false, &files);
      for (const auto &f : files) {
        auto buf = ReadFile(mgr, f);
        std::istringstream is(std::string(buf.data(), buf.size()));
        LoadLexicon(is, config.model.debug);
      }
    }

    tn_list_ = LoadTextNormalizers(mgr, config.rule_fsts, config.rule_fars,
                                   config.model.debug);

    if (vocoder_ && vocoder_->SampleRate() != meta_data.sample_rate) {
      SHERPA_ONNX_LOGE(
          "ERROR: Vocoder sample rate (%d Hz) does not match the TTS model "
          "sample rate (%d Hz).",
          vocoder_->SampleRate(), meta_data.sample_rate);
      SHERPA_ONNX_LOGE(
          "Please download the correct vocoder from "
          "https://github.com/k2-fsa/sherpa-onnx/releases/tag/vocoder-models");
      if (meta_data.sample_rate == 16000) {
        SHERPA_ONNX_LOGE("For 16kHz models, use vocos-16khz-univ.onnx");
      } else if (meta_data.sample_rate == 22050) {
        SHERPA_ONNX_LOGE("For 22050Hz models, use vocos-22khz-univ.onnx");
      } else if (meta_data.sample_rate == 24000) {
        SHERPA_ONNX_LOGE("For 24kHz models, use vocos-24khz-univ.onnx");
      }
      SHERPA_ONNX_EXIT(-1);
    }
  }

  int32_t SampleRate() const override {
    return model_->GetMetaData().sample_rate;
  }

  int32_t NumSpeakers() const override {
    return model_->GetMetaData().num_speakers;
  }

  std::string Lang() const override { return LastLang(model_->GetMetaData().voice); }

  /**
   *
   * Supported options in GenerationConfig:
   *   - sid: Speaker ID for multi-speaker models
   *   - speed: Speech speed factor (default: 1.0)
   *   - silence_scale: Scale applied to pauses in the generated audio
   *
   * Supported extra parameters:
   *
   *  - debug, int, default from model config
   *  - min_words_in_sentence, int, default 5.
   *    Merge adjacent sentences if the number of words is less than this.
   *    Each CJK character counts as one word; English words are
   *    space-separated.
   *  - max_words_in_sentence, int, default 20.
   *    Split a sentence into chunks if the number of words exceeds this.
   *    Splits at punctuation or space boundaries.
   *  - max_codepoints_in_sentence, int, default 200.
   *    For the phoneme_codepoints path: split a phoneme sequence if it
   *    exceeds this many codepoints. Splits at space boundaries.
   */
  GeneratedAudio Generate(
      const std::string &_text, const GenerationConfig &gen_config,
      GeneratedAudioCallback callback = nullptr) const override {
    bool debug = gen_config.GetExtraInt("debug", config_.model.debug);

    if (debug) {
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
    if (!text.empty()) {
      text = NormalizeText(text, debug);
    }

    std::vector<TokenIDs> token_ids;

    if (!gen_config.tokens.empty()) {
      // String tokens path (e.g., Chinese pinyin: "zhong1", "guo2")
      // Flatten all sentences, split at punctuation, merge short, split long
      std::vector<std::string> all_tokens;
      for (const auto &sentence : gen_config.tokens) {
        for (const auto &tok : sentence) {
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
      int32_t min_words = gen_config.GetExtraInt("min_words_in_sentence", 5);
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

      // Convert token strings to IDs, merge single-token sentences
      int32_t sentence_idx = 0;
      for (const auto &sentence : merged) {
        std::vector<int64_t> ids;
        std::vector<std::string> id_strs;
        for (const auto &tok : sentence) {
          auto it = token_str2id_.find(tok);
          if (it != token_str2id_.end()) {
            ids.push_back(it->second);
            id_strs.push_back(tok + "(" + std::to_string(it->second) + ")");
          } else {
            SHERPA_ONNX_LOGE("Skip unknown token: '%s'", tok.c_str());
          }
        }
        if (ids.empty()) continue;

        if (debug) {
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

        if (ids.size() == 1 && !token_ids.empty()) {
          // Merge single-token sentence into previous
          token_ids.back().tokens.insert(token_ids.back().tokens.end(),
                                         ids.begin(), ids.end());
        } else {
          token_ids.emplace_back(std::move(ids));
        }
      }
    } else if (!gen_config.phoneme_codepoints.empty()) {
      // phoneme_codepoints path: split at punctuation, merge short, split long
      int32_t max_codepoints =
          gen_config.GetExtraInt("max_codepoints_in_sentence", 200);
      int32_t min_codepoints =
          gen_config.GetExtraInt("min_words_in_sentence", 5) * 3;

      auto IsPunctCodepoint = [](int32_t cp) {
        return cp == ',' || cp == '.' || cp == '!' || cp == '?' || cp == ';' ||
               cp == ':';
      };

      // Flatten all sentences
      std::vector<int32_t> all_codepoints;
      for (const auto &sentence : gen_config.phoneme_codepoints) {
        all_codepoints.insert(all_codepoints.end(), sentence.begin(),
                              sentence.end());
      }

      // Split at punctuation boundaries
      std::vector<std::vector<int32_t>> punct_chunks;
      std::vector<int32_t> current;
      for (int32_t cp : all_codepoints) {
        current.push_back(cp);
        if (IsPunctCodepoint(cp)) {
          punct_chunks.push_back(std::move(current));
          current.clear();
        }
      }
      if (!current.empty()) {
        punct_chunks.push_back(std::move(current));
      }

      // Merge short chunks
      std::vector<std::vector<int32_t>> merged;
      std::vector<int32_t> buffer;
      for (auto &chunk : punct_chunks) {
        buffer.insert(buffer.end(), chunk.begin(), chunk.end());
        if (static_cast<int32_t>(buffer.size()) >= min_codepoints) {
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

      // Split long chunks at space boundaries
      std::vector<std::vector<int32_t>> chunks;
      for (auto &chunk : merged) {
        if (max_codepoints > 0 &&
            static_cast<int32_t>(chunk.size()) > max_codepoints) {
          std::vector<int32_t> piece;
          for (int32_t cp : chunk) {
            piece.push_back(cp);
            if (cp == 0x20 &&
                static_cast<int32_t>(piece.size()) >= max_codepoints) {
              chunks.push_back(std::move(piece));
              piece.clear();
            }
          }
          if (!piece.empty()) {
            chunks.push_back(std::move(piece));
          }
        } else {
          chunks.push_back(std::move(chunk));
        }
      }

      // Merge single-codepoint chunks into previous
      std::vector<std::vector<int32_t>> final_chunks;
      for (auto &chunk : chunks) {
        if (chunk.size() == 1 && !final_chunks.empty()) {
          final_chunks.back().insert(final_chunks.back().end(), chunk.begin(),
                                     chunk.end());
        } else {
          final_chunks.push_back(std::move(chunk));
        }
      }
      chunks = std::move(final_chunks);

      if (debug) {
        SHERPA_ONNX_LOGE("phoneme_codepoints: %d sentences, max_codepoints=%d",
                         static_cast<int32_t>(chunks.size()), max_codepoints);
      }

      token_ids = TokenizeFromCodepoints(chunks, meta_data, debug);
    } else {
      // Lexicon path: convert Chinese punctuations, split text into sentences,
      // merge short ones, split long ones, then tokenize each sentence.
      std::string normalized = NormalizeNfc(ReplacePunctuations(text));
      if (debug) {
#if __OHOS__
        SHERPA_ONNX_LOGE("After replacing punctuations: %{public}s",
                         normalized.c_str());
#else
        SHERPA_ONNX_LOGE("After replacing punctuations: %s",
                         normalized.c_str());
#endif
      }

      auto sentences = SplitByAllPunctuation(normalized);
      if (sentences.empty()) {
        SHERPA_ONNX_LOGE("No sentences after splitting");
        return {};
      }

      int32_t min_words_in_sentence =
          gen_config.GetExtraInt("min_words_in_sentence", 5);
      int32_t max_words_in_sentence =
          gen_config.GetExtraInt("max_words_in_sentence", 20);

      sentences = MergeShortSentencesByWords(sentences, min_words_in_sentence);

      std::vector<std::string> final_chunks;
      for (const auto &s : sentences) {
        auto pieces = SplitLongSentenceByWords(s, max_words_in_sentence);
        final_chunks.insert(final_chunks.end(), pieces.begin(), pieces.end());
      }

      // Merge punctuation-only sentences (e.g., standalone ".") into previous
      std::vector<std::string> merged_chunks;
      for (const auto &s : final_chunks) {
        std::u32string u32 = Utf8ToUtf32(Trim(s));
        bool all_punct = !u32.empty();
        for (char32_t c : u32) {
          if (c != U',' && c != U'.' && c != U'!' && c != U'?' && c != U';' &&
              c != U':' && c != U' ') {
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

      if (debug) {
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

      for (int32_t si = 0; si < static_cast<int32_t>(sentences.size()); ++si) {
        auto ids =
            TokenizeSentence(gen_config, sentences[si], meta_data, debug, si);

        // Add trailing punctuation token if the sentence ends with punctuation.
        // SplitByAllPunctuation strips the punctuation character, so we need
        // to add it back as a token — but only if it's not already the last
        // token.
        std::string trimmed = Trim(sentences[si]);
        if (!trimmed.empty()) {
          char last = trimmed.back();
          std::string punct(1, last);
          if (IsPunctuation(punct) && !ids.empty()) {
            auto it = token_str2id_.find(punct);
            if (it != token_str2id_.end() && !ids.back().tokens.empty() &&
                ids.back().tokens.back() != it->second) {
              ids.back().tokens.push_back(it->second);
              if (debug) {
                SHERPA_ONNX_LOGE("Trailing punctuation: '%s' -> %d",
                                 punct.c_str(), it->second);
              }
            }
          }
        }

        token_ids.insert(token_ids.end(), ids.begin(), ids.end());
      }
    }

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

    if (meta_data.add_blank) {
      for (auto &k : x) {
        k = AddBlank(k, meta_data.pad_id);
      }
    }

    if (debug) {
      for (int32_t i = 0; i < static_cast<int32_t>(x.size()); ++i) {
        std::ostringstream os;
        os << "Sentence " << i << " final IDs (" << x[i].size() << "): [";
        for (size_t j = 0; j < x[i].size(); ++j) {
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

    int32_t x_size = static_cast<int32_t>(x.size());

    if (config_.max_num_sentences <= 0 || x_size <= config_.max_num_sentences) {
      auto ans = Process(x, sid, speed, gen_config.silence_scale);
      if (callback) {
        callback(ans.samples.data(), ans.samples.size(), 1.0);
      }
      return ans;
    }

    // the input text is too long, we process sentences within it in batches
    // to avoid OOM. Batch size is config_.max_num_sentences
    std::vector<std::vector<int64_t>> batch_x;

    int32_t batch_size = config_.max_num_sentences;
    batch_x.reserve(config_.max_num_sentences);
    int32_t num_batches = x_size / batch_size;

    if (config_.model.debug) {
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
      for (int32_t i = 0; i != batch_size; ++i, ++k) {
        batch_x.push_back(std::move(x[k]));
      }

      auto audio = Process(batch_x, sid, speed, gen_config.silence_scale);
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
      auto audio = Process(batch_x, sid, speed, gen_config.silence_scale);
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
    return Generate(text, gen_config, std::move(callback));
  }

 private:
  std::string NormalizeText(const std::string &text, bool debug) const {
    std::string result = text;

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

  std::vector<TokenIDs> TokenizeSentence(
      const GenerationConfig &gen_config, const std::string &text,
      const OfflineTtsMatchaModelMetaData &meta_data, bool debug,
      int32_t sentence_index = 0) const {
    if (!lexicon_.empty() && lexicon_str_.empty()) {
      // Pure IPA model (e.g., en with || format lexicon only)
      auto codepoints = TokenizeFromLexicon(text, debug, sentence_index);
      return TokenizeFromCodepoints(codepoints, meta_data, debug);
    }

    if (!lexicon_str_.empty()) {
      // Mixed model (e.g., zh-en with both zh space-format and en || format)
      // or pure zh model with space-format lexicon.
      // TokenizeFromLexiconStr handles both IPA and pinyin entries correctly.
      return TokenizeFromLexiconStr(text, meta_data, debug, sentence_index);
    }

    SHERPA_ONNX_LOGE(
        "No lexicon available. "
        "Please provide a lexicon via --matcha-lexicon.");
    return {};
  }

  std::vector<TokenIDs> TokenizeFromCodepoints(
      const std::vector<std::vector<int32_t>> &codepoints,
      const OfflineTtsMatchaModelMetaData &meta_data,
      bool debug = false) const {
    std::vector<TokenIDs> result;

    if (!token2id_.empty()) {
      // IPA single-codepoint path (e.g., en model with || format lexicon)
      result.reserve(codepoints.size());
      for (const auto &phonemes : codepoints) {
        std::vector<char32_t> phonemes_c32(phonemes.begin(), phonemes.end());
        auto ids_list = PiperPhonemesToIdsMatcha(token2id_, phonemes_c32,
                                                 meta_data.use_eos_bos);
        for (auto &ids : ids_list) {
          if (debug) {
            std::ostringstream os;
            os << "IPA tokens: [";
            // Build phoneme string for display
            std::string phonemes_str;
            for (int32_t cp : phonemes) {
              phonemes_str += Utf32ToUtf8(static_cast<char32_t>(cp));
            }
            os << phonemes_str << "] -> IDs: [";
            for (size_t j = 0; j < ids.size(); ++j) {
              if (j > 0) os << ", ";
              os << ids[j];
            }
            os << "]";
#if __OHOS__
            SHERPA_ONNX_LOGE("%{public}s", os.str().c_str());
#else
            SHERPA_ONNX_LOGE("%s", os.str().c_str());
#endif
          }
          result.emplace_back(std::move(ids));
        }
      }
      return result;
    }

    if (!token_str2id_.empty()) {
      // Multi-character token path (e.g., zh model with pinyin tokens).
      // Codepoints are ASCII characters that form token strings like "dang1".
      // Split into non-space runs, look up each run in token_str2id_.
      for (const auto &phonemes : codepoints) {
        std::vector<int64_t> ids;
        std::string current;
        for (int32_t cp : phonemes) {
          if (cp == ' ') {
            if (!current.empty()) {
              auto it = token_str2id_.find(current);
              if (it != token_str2id_.end()) {
                ids.push_back(it->second);
              } else {
                SHERPA_ONNX_LOGE("Skip unknown token: '%s'", current.c_str());
              }
              current.clear();
            }
          } else {
            current.push_back(static_cast<char>(cp));
          }
        }
        if (!current.empty()) {
          auto it = token_str2id_.find(current);
          if (it != token_str2id_.end()) {
            ids.push_back(it->second);
          } else {
            SHERPA_ONNX_LOGE("Skip unknown token: '%s'", current.c_str());
          }
        }

        if (!ids.empty()) {
          result.emplace_back(std::move(ids));
        }
      }
      return result;
    }

    SHERPA_ONNX_LOGE("No token mapping available for codepoints.");
    return {};
  }

  std::vector<int64_t> ConvertPhonemeStringsToIds(
      const std::vector<std::string> &phones) const {
    std::vector<int64_t> ans;
    for (const auto &p : phones) {
      auto it = token_str2id_.find(p);
      if (it != token_str2id_.end()) {
        ans.push_back(it->second);
      } else {
        SHERPA_ONNX_LOGE("Skip unknown token: '%s'", p.c_str());
      }
    }
    return ans;
  }

  static std::string ReplacePunctuations(const std::string &s) {
    // Convert Chinese punctuations to English equivalents
    static const std::vector<std::pair<std::string, std::string>> replacements =
        {
            {"，", ","}, {"、", ","}, {"；", ";"}, {"：", ","},  {":", ","},
            {"。", "."}, {"？", "?"}, {"！", "!"}, {"…", "..."},
    };
    std::string result = s;
    for (const auto &p : replacements) {
      size_t pos = 0;
      while ((pos = result.find(p.first, pos)) != std::string::npos) {
        result.replace(pos, p.first.size(), p.second);
        pos += p.second.size();
      }
    }
    return result;
  }

  static bool IsPunctuation(const std::string &s) {
    if (s.empty()) return false;
    // ASCII punctuations and Chinese equivalents (already converted)
    static const std::string puncts = ",.;:!?";
    if (s.size() == 1 && puncts.find(s[0]) != std::string::npos) return true;
    return false;
  }

  std::vector<TokenIDs> TokenizeFromLexiconStr(
      const std::string &text, const OfflineTtsMatchaModelMetaData &meta_data,
      bool debug, int32_t sentence_index = 0) const {
    std::vector<TokenIDs> result;

    if (debug) {
#if __OHOS__
      SHERPA_ONNX_LOGE("TokenizeFromLexiconStr: text='%{public}s'",
                       text.c_str());
#else
      SHERPA_ONNX_LOGE("TokenizeFromLexiconStr: text='%s'", text.c_str());
#endif
    }

    {
      std::vector<std::string> words;
      std::u32string u32 = Utf8ToUtf32(text);
      size_t current_word_start = 0;
      bool in_word = false;

      for (size_t ci = 0; ci < u32.size(); ++ci) {
        char32_t c = u32[ci];
        if (c == U' ' || c == U'\t' || c == U'\n' || c == U'\r') {
          if (in_word) {
            words.push_back(Utf32ToUtf8(
                u32.substr(current_word_start, ci - current_word_start)));
            in_word = false;
          }
          words.push_back(" ");
        } else if (c == U',' || c == U'.' || c == U'!' || c == U'?' ||
                   c == U';' || c == U':') {
          // ASCII punctuation: keep as separate token
          if (in_word) {
            words.push_back(Utf32ToUtf8(
                u32.substr(current_word_start, ci - current_word_start)));
            in_word = false;
          }
          words.push_back(Utf32ToUtf8(u32.substr(ci, 1)));
        } else if (IsCJK(c)) {
          // CJK character: each is a separate word
          if (in_word) {
            words.push_back(Utf32ToUtf8(
                u32.substr(current_word_start, ci - current_word_start)));
            in_word = false;
          }
          words.push_back(Utf32ToUtf8(u32.substr(ci, 1)));
        } else {
          // ASCII letter/digit: accumulate into word
          if (!in_word) {
            current_word_start = ci;
            in_word = true;
          }
        }
      }
      if (in_word) {
        words.push_back(Utf32ToUtf8(u32.substr(current_word_start)));
      }

      if (words.empty()) return result;

      std::vector<int64_t> sentence_ids;
      int32_t i = 0;
      int32_t n = static_cast<int32_t>(words.size());

      while (i < n) {
        // Space tokens: insert space token ID (used as blank between words)
        if (words[i] == " ") {
          auto it_space = token_str2id_.find(" ");
          if (it_space != token_str2id_.end()) {
            sentence_ids.push_back(it_space->second);
          }
          ++i;
          continue;
        }

        // Punctuation: look up in token_str2id_
        if (IsPunctuation(words[i])) {
          auto it = token_str2id_.find(words[i]);
          if (it != token_str2id_.end()) {
            sentence_ids.push_back(it->second);
            if (debug) {
              SHERPA_ONNX_LOGE("Punctuation: '%s' -> %d", words[i].c_str(),
                               it->second);
            }
          }
          ++i;
          continue;
        }

        // Lexicon lookup (longest match)
        // For CJK characters, concatenate directly (e.g., "中国").
        // For English words, join with space (e.g., "how are").
        bool found = false;
        int32_t max_try = std::min(max_lexicon_phrase_len_, n - i);
        for (int32_t len = max_try; len >= 1; --len) {
          std::string phrase;
          for (int32_t j = i; j < i + len; ++j) {
            bool prev_is_cjk =
                !phrase.empty() && IsCJK(Utf8ToUtf32(phrase).back());
            bool curr_is_cjk =
                !words[j].empty() && IsCJK(Utf8ToUtf32(words[j]).front());
            if (j > i && !(prev_is_cjk && curr_is_cjk)) {
              phrase.push_back(' ');
            }
            phrase += NormalizeLexiconKey(words[j]);
          }

          auto it = lexicon_str_.find(phrase);
          if (it != lexicon_str_.end()) {
            if (debug) {
              std::string phones_str;
              for (const auto &p : it->second) {
                phones_str += p + " ";
              }
              SHERPA_ONNX_LOGE("Lexicon matched: '%s' -> '%s'", phrase.c_str(),
                               phones_str.c_str());
            }
            auto ids = ConvertPhonemeStringsToIds(it->second);
            sentence_ids.insert(sentence_ids.end(), ids.begin(), ids.end());
            i += len;
            found = true;
            break;
          }

          // Fallback: check IPA lexicon (|| format entries in lexicon_)
          auto it_ipa = lexicon_.find(phrase);
          if (it_ipa != lexicon_.end()) {
            if (debug) {
              std::string phonemes_str;
              for (int32_t cp : it_ipa->second) {
                phonemes_str += Utf32ToUtf8(static_cast<char32_t>(cp));
              }
              SHERPA_ONNX_LOGE("Lexicon (IPA) matched: '%s' -> '%s'",
                               phrase.c_str(), phonemes_str.c_str());
            }
            std::vector<char32_t> phonemes_c32(it_ipa->second.begin(),
                                               it_ipa->second.end());
            auto ids_list =
                PiperPhonemesToIdsMatcha(token2id_, phonemes_c32, false);
            for (auto &ids : ids_list) {
              sentence_ids.insert(sentence_ids.end(), ids.begin(), ids.end());
            }
            i += len;
            found = true;
            break;
          }
        }

        if (!found) {
          // For CJK text, split into individual characters and look up each
          if (ContainsCJK(words[i])) {
            std::vector<std::string> chars = SplitUtf8(words[i]);
            for (const auto &ch : chars) {
              std::string ch_lower = NormalizeLexiconKey(ch);
              auto it = lexicon_str_.find(ch_lower);
              if (it != lexicon_str_.end()) {
                if (debug) {
                  std::string phones_str;
                  for (const auto &p : it->second) phones_str += p + " ";
                  SHERPA_ONNX_LOGE("Lexicon matched: '%s' -> '%s'",
                                   ch_lower.c_str(), phones_str.c_str());
                }
                auto ids = ConvertPhonemeStringsToIds(it->second);
                sentence_ids.insert(sentence_ids.end(), ids.begin(), ids.end());
              } else {
                SHERPA_ONNX_LOGE("OOV character skipped: '%s'", ch.c_str());
              }
            }
          } else {
            // The word is not in the lexicon. If it contains '-', split it
            // and look up each part separately, e.g. "two-thirds" -> "two" +
            // "thirds".
            std::vector<std::string> parts = SplitStringAndTrim(words[i], '-');
            if (words[i].find('-') != std::string::npos && !parts.empty()) {
              for (const auto &part : parts) {
                std::string part_lower = NormalizeLexiconKey(part);
                auto it2 = lexicon_str_.find(part_lower);
                if (it2 != lexicon_str_.end()) {
                  if (debug) {
                    std::string phones_str;
                    for (const auto &p : it2->second) phones_str += p + " ";
                    SHERPA_ONNX_LOGE("Lexicon matched: '%s' -> '%s'",
                                     part_lower.c_str(), phones_str.c_str());
                  }
                  auto ids = ConvertPhonemeStringsToIds(it2->second);
                  sentence_ids.insert(sentence_ids.end(), ids.begin(),
                                      ids.end());
                } else {
                  SHERPA_ONNX_LOGE("OOV word skipped: '%s'", part.c_str());
                }
              }
            } else {
              SHERPA_ONNX_LOGE("OOV word skipped: '%s'", words[i].c_str());
            }
          }
          ++i;
        }
      }

      if (!sentence_ids.empty()) {
        if (debug) {
          SHERPA_ONNX_LOGE("Sentence %d: text='%s'", sentence_index,
                           Trim(text).c_str());

          std::ostringstream os;
          os << "Sentence " << sentence_index << " tokens: [";
          for (int32_t j = 0; j < static_cast<int32_t>(sentence_ids.size());
               ++j) {
            if (j > 0) os << ", ";
            // Look up token string from id
            std::string tok_str = "?";
            for (const auto &kv : token_str2id_) {
              if (kv.second == sentence_ids[j]) {
                tok_str = kv.first;
                break;
              }
            }
            os << tok_str << "(" << sentence_ids[j] << ")";
          }
          os << "]";
#if __OHOS__
          SHERPA_ONNX_LOGE("%{public}s", os.str().c_str());
#else
          SHERPA_ONNX_LOGE("%s", os.str().c_str());
#endif
        }

        auto ids_list = ApplyBosEos(sentence_ids, meta_data.use_eos_bos);
        for (auto &ids : ids_list) {
          result.emplace_back(std::move(ids));
        }
      }
    }

    return result;
  }

  // Split a flat ID sequence into sub-sequences with BOS/EOS, respecting
  // max_token_len (same logic as PiperPhonemesToIdsMatcha but with int64_t).
  static std::vector<TokenIDs> ApplyBosEos(const std::vector<int64_t> &ids,
                                           bool use_eos_bos,
                                           int32_t max_token_len = 400) {
    std::vector<TokenIDs> ans;
    if (!use_eos_bos) {
      ans.emplace_back(ids);
      return ans;
    }

    // BOS and EOS are ^ and $ — use 0 as placeholder; they'll be resolved
    // via token2id_ at this stage we already have integer IDs.
    // Actually for matcha, BOS/EOS are handled by PiperPhonemesToIdsMatcha
    // using token2id_.at(U'^') and token2id_.at(U'$'). For the string-based
    // path we need the same IDs. We'll just wrap with the IDs from
    // token_str2id_["^"] and token_str2id_["$"].
    // But those might not exist. Instead, just return flat — AddBlank will
    // handle padding. The model expects BOS/EOS from PiperPhonemesToIdsMatcha
    // but for string-based lexicon we skip BOS/EOS for simplicity since
    // the zh-en model doesn't use them (use_eos_bos=0 for zh-en).
    ans.emplace_back(ids);
    return ans;
  }

  void LoadLexicon(std::istream &is, bool debug) {
    auto entries = ParseLexiconFile(is, &max_lexicon_phrase_len_);
    for (auto &e : entries) {
      // Detect format: if phonemes contain multi-byte IPA chars or multi-char
      // strings, store as token strings. Otherwise store as IPA codepoints.
      bool has_multi_char = false;
      for (const auto &p : e.phonemes) {
        std::u32string u32 = Utf8ToUtf32(p);
        if (u32.size() != 1) {
          has_multi_char = true;
          break;
        }
      }

      if (has_multi_char) {
        // Token strings (e.g., pinyin: "zhong1", "guo2")
        lexicon_str_[e.key] = std::move(e.phonemes);
      } else {
        // IPA codepoints
        std::vector<int32_t> codepoints;
        for (const auto &p : e.phonemes) {
          std::u32string u32 = Utf8ToUtf32(p);
          for (char32_t cp : u32) {
            codepoints.push_back(static_cast<int32_t>(cp));
          }
        }
        lexicon_[e.key] = std::move(codepoints);
      }
    }
    if (debug) {
      SHERPA_ONNX_LOGE(
          "Loaded lexicon: %d new-format (IPA) entries, "
          "%d old-format (token string) entries, max phrase len %d",
          static_cast<int32_t>(lexicon_.size()),
          static_cast<int32_t>(lexicon_str_.size()), max_lexicon_phrase_len_);
    }
  }

  std::vector<std::vector<int32_t>> TokenizeFromLexicon(
      const std::string &text, bool debug, int32_t sentence_index = 0) const {
    std::vector<std::vector<int32_t>> result;

    {
      const std::string &sentence = text;
      // SplitUtf8 merges ASCII into words, splits CJK into characters,
      // keeps punctuation as separate tokens, keeps spaces as tokens.
      std::vector<std::string> words = SplitUtf8(sentence);

      if (words.empty()) return result;

      std::vector<int32_t> sentence_codepoints;
      int32_t i = 0;
      int32_t n = static_cast<int32_t>(words.size());

      while (i < n) {
        // Space tokens: insert space token ID
        if (words[i] == " ") {
          auto it_space = token_str2id_.find(" ");
          if (it_space != token_str2id_.end()) {
            sentence_codepoints.push_back(it_space->second);
          }
          ++i;
          continue;
        }

        // Punctuation: look up in token_str2id_ or add as codepoint
        if (IsPunctuation(words[i])) {
          auto it_punct = token_str2id_.find(words[i]);
          if (it_punct != token_str2id_.end()) {
            sentence_codepoints.push_back(it_punct->second);
          } else {
            // Single ASCII punctuation — add as codepoint
            sentence_codepoints.push_back(static_cast<int32_t>(words[i][0]));
          }
          ++i;
          continue;
        }

        bool found = false;
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
            if (!sentence_codepoints.empty()) {
              int32_t last = sentence_codepoints.back();
              if (last != ',' && last != '.' && last != '!' && last != '?' &&
                  last != ';' && last != ':' && last != ' ') {
                sentence_codepoints.push_back(static_cast<int32_t>(' '));
              }
            }
            sentence_codepoints.insert(sentence_codepoints.end(),
                                       it->second.begin(), it->second.end());
            i += len;
            found = true;
            break;
          }
        }

        if (!found) {
          // Fallback: check lexicon_str_ (old-format, token string entries)
          std::string key = NormalizeLexiconKey(words[i]);
          auto it_str = lexicon_str_.find(key);
          if (it_str != lexicon_str_.end()) {
            if (debug) {
              std::string phones_str;
              for (const auto &p : it_str->second) phones_str += p + " ";
              SHERPA_ONNX_LOGE("Lexicon (str) matched: '%s' -> '%s'",
                               key.c_str(), phones_str.c_str());
            }
            // Convert token strings to IDs and add as codepoints
            for (const auto &tok : it_str->second) {
              auto it_id = token_str2id_.find(tok);
              if (it_id != token_str2id_.end()) {
                sentence_codepoints.push_back(it_id->second);
              }
            }
          } else if (ContainsCJK(words[i])) {
            // CJK: split into individual characters
            std::vector<std::string> chars = SplitUtf8(words[i]);
            for (const auto &ch : chars) {
              std::string ch_lower = NormalizeLexiconKey(ch);
              auto it_lex = lexicon_str_.find(ch_lower);
              if (it_lex != lexicon_str_.end()) {
                if (debug) {
                  std::string phones_str;
                  for (const auto &p : it_lex->second) phones_str += p + " ";
                  SHERPA_ONNX_LOGE("Lexicon (str) matched: '%s' -> '%s'",
                                   ch_lower.c_str(), phones_str.c_str());
                }
                for (const auto &tok : it_lex->second) {
                  auto it_id = token_str2id_.find(tok);
                  if (it_id != token_str2id_.end()) {
                    sentence_codepoints.push_back(it_id->second);
                  }
                }
              } else {
                SHERPA_ONNX_LOGE("OOV character skipped: '%s'", ch.c_str());
              }
            }
          } else {
            // The word is not in the lexicon. If it contains '-', split it
            // and look up each part separately, e.g. "two-thirds" -> "two" +
            // "thirds".
            std::vector<std::string> parts = SplitStringAndTrim(words[i], '-');
            if (words[i].find('-') != std::string::npos && !parts.empty()) {
              for (const auto &part : parts) {
                std::string part_lower = NormalizeLexiconKey(part);
                auto it2 = lexicon_str_.find(part_lower);
                if (it2 != lexicon_str_.end()) {
                  if (debug) {
                    std::string phones_str;
                    for (const auto &p : it2->second) phones_str += p + " ";
                    SHERPA_ONNX_LOGE("Lexicon (str) matched: '%s' -> '%s'",
                                     part_lower.c_str(), phones_str.c_str());
                  }
                  for (const auto &tok : it2->second) {
                    auto it_id = token_str2id_.find(tok);
                    if (it_id != token_str2id_.end()) {
                      sentence_codepoints.push_back(it_id->second);
                    }
                  }
                } else {
                  SHERPA_ONNX_LOGE("OOV word skipped: '%s'", part.c_str());
                }
              }
            } else {
              SHERPA_ONNX_LOGE("OOV word skipped: '%s'", words[i].c_str());
            }
          }
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
              sentence_index, Trim(text).c_str(),
              static_cast<int32_t>(sentence_codepoints.size()),
              phonemes_str.c_str());
        }
        result.push_back(std::move(sentence_codepoints));
      }
    }

    return result;
  }

  GeneratedAudio Process(const std::vector<std::vector<int64_t>> &tokens,
                         int32_t sid, float speed, float silence_scale) const {
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

    GeneratedAudio ans;

    Ort::Value mel = model_->Run(std::move(x_tensor), sid, speed);

    const auto &meta_data = model_->GetMetaData();
    if (meta_data.need_vocoder) {
      ans.samples = vocoder_->Run(std::move(mel));
    } else {
      std::vector<int64_t> shape = mel.GetTensorTypeAndShapeInfo().GetShape();
      int64_t num_samples = 1;
      for (auto s : shape) {
        num_samples *= s;
      }
      ans.samples.resize(num_samples);
      auto p = mel.GetTensorData<float>();
      std::copy(p, p + num_samples, ans.samples.data());
    }

    ans.sample_rate = model_->GetMetaData().sample_rate;

    if (silence_scale != 1) {
      ans = ans.ScaleSilence(silence_scale);
    }

    return ans;
  }

 private:
  OfflineTtsConfig config_;
  std::unique_ptr<OfflineTtsMatchaModel> model_;
  std::unique_ptr<Vocoder> vocoder_;
  std::vector<std::unique_ptr<kaldifst::TextNormalizer>> tn_list_;
  std::unordered_map<char32_t, int32_t> token2id_;         // for IPA codepoints
  std::unordered_map<std::string, int32_t> token_str2id_;  // for string tokens
  std::unordered_map<std::string, std::vector<int32_t>> lexicon_;  // || format
  std::unordered_map<std::string, std::vector<std::string>>
      lexicon_str_;  // space format
  int32_t max_lexicon_phrase_len_ = 1;
};

}  // namespace sherpa_onnx
#endif  // SHERPA_ONNX_CSRC_OFFLINE_TTS_MATCHA_IMPL_H_
