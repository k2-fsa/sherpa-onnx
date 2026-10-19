// sherpa-onnx/csrc/offline-tts-kokoro-impl.h
//
// Copyright (c)  2025  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_OFFLINE_TTS_KOKORO_IMPL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_TTS_KOKORO_IMPL_H_

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
#include "sherpa-onnx/csrc/offline-tts-kokoro-model.h"
#include "sherpa-onnx/csrc/symbol-table.h"
#include "sherpa-onnx/csrc/text-utils.h"
#include "sherpa-onnx/csrc/tts-text-normalizer.h"

namespace sherpa_onnx {

class OfflineTtsKokoroImpl : public OfflineTtsImpl {
 public:
  explicit OfflineTtsKokoroImpl(const OfflineTtsConfig &config)
      : config_(config),
        model_(std::make_unique<OfflineTtsKokoroModel>(config.model)) {
    if (!config.model.kokoro.tokens.empty()) {
      auto is = OpenInputFile(config.model.kokoro.tokens);
      token_str2id_ = ReadTokens(is);

      // Also load char32_t → int32_t map for IPA single-codepoint tokens
      auto is2 = OpenInputFile(config.model.kokoro.tokens);
      std::string line;
      while (std::getline(is2, line)) {
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
          token2id_[u32[0]] = id;
        }
      }
    }

    if (!config.model.kokoro.lexicon.empty()) {
      std::vector<std::string> files;
      SplitStringToVector(config.model.kokoro.lexicon, ",", false, &files);
      for (const auto &f : files) {
        auto is = OpenInputFile(f);
        LoadLexicon(is, config.model.debug);
      }
    }

    tn_list_ = LoadTextNormalizers(config.rule_fsts, config.rule_fars,
                                   config.model.debug);
  }

  template <typename Manager>
  OfflineTtsKokoroImpl(Manager *mgr, const OfflineTtsConfig &config)
      : config_(config),
        model_(std::make_unique<OfflineTtsKokoroModel>(mgr, config.model)) {
    if (!config.model.kokoro.tokens.empty()) {
      auto buf = ReadFile(mgr, config.model.kokoro.tokens);
      std::istringstream is(std::string(buf.data(), buf.size()));
      token_str2id_ = ReadTokens(is);

      auto buf2 = ReadFile(mgr, config.model.kokoro.tokens);
      std::istringstream is2(std::string(buf2.data(), buf2.size()));
      std::string line;
      while (std::getline(is2, line)) {
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
          token2id_[u32[0]] = id;
        }
      }
    }

    if (!config.model.kokoro.lexicon.empty()) {
      std::vector<std::string> files;
      SplitStringToVector(config.model.kokoro.lexicon, ",", false, &files);
      for (const auto &f : files) {
        auto buf = ReadFile(mgr, f);
        std::istringstream is(std::string(buf.data(), buf.size()));
        LoadLexicon(is, config.model.debug);
      }
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

  /**
   *
   * Supported options in GenerationConfig:
   *   - sid: Speaker ID for multi-speaker models
   *   - speed: Speech speed factor (default: 1.0)
   *   - silence_scale: Scale applied to pauses in the generated audio
   *   - tokens: Vector of sentences, each a vector of token strings.
   *     Takes precedence over text when provided.
   *
   * Supported extra parameters:
   *
   *  - debug, int, default from model config
   *  - min_words_in_sentence, int, default 5.
   *    Merge adjacent sentences if the number of words is less than this.
   *    Each CJK character counts as one word; English words are
   * space-separated.
   *  - max_words_in_sentence, int, default 20.
   *    Split a sentence into chunks if the number of words exceeds this.
   *    Splits at punctuation or space boundaries.
   *  - lang, str, default from model config.
   *    Language override for sentence splitting.
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
      // tokens path: split at punctuation, merge short, split long
      // Flatten all tokens
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

      // Merge single-token sentences into previous
      std::vector<std::vector<std::string>> final_sentences;
      for (auto &ts : merged) {
        if (ts.size() == 1 && !final_sentences.empty()) {
          final_sentences.back().insert(final_sentences.back().end(),
                                        ts.begin(), ts.end());
        } else {
          final_sentences.push_back(std::move(ts));
        }
      }

      // Convert token strings to IDs, wrap with PAD
      int32_t si = 0;
      for (const auto &sentence : final_sentences) {
        std::vector<int64_t> ids;
        ids.push_back(0);  // PAD at start
        for (const auto &tok : sentence) {
          auto it = token_str2id_.find(tok);
          if (it != token_str2id_.end()) {
            ids.push_back(it->second);
          } else {
            // Multi-character token not found as whole — split into
            // individual characters and look up each one.
            std::u32string u32 = Utf8ToUtf32(tok);
            bool found_sub = false;
            for (char32_t c : u32) {
              std::string ch = Utf32ToUtf8(std::u32string(1, c));
              auto it2 = token_str2id_.find(ch);
              if (it2 != token_str2id_.end()) {
                ids.push_back(it2->second);
                found_sub = true;
              }
            }
            if (!found_sub && debug) {
              SHERPA_ONNX_LOGE("Skip unknown token: '%s'", tok.c_str());
            }
          }
        }
        ids.push_back(0);  // PAD at end

        if (debug) {
          std::ostringstream os;
          os << "Sentence " << si << " tokens: [";
          for (size_t j = 0; j < ids.size(); ++j) {
            if (j > 0) os << ", ";
            // Look up token string from id
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

        if (ids.size() > 2) {  // more than just PAD + PAD
          token_ids.emplace_back(std::move(ids));
        }
        ++si;
      }
    } else if (!gen_config.phoneme_codepoints.empty()) {
      // phoneme_codepoints path
      int32_t si = 0;
      for (const auto &sentence : gen_config.phoneme_codepoints) {
        std::vector<char32_t> phonemes(sentence.begin(), sentence.end());
        auto ids_list = PiperPhonemesToIdsMatcha(token2id_, phonemes, false);
        for (auto &ids : ids_list) {
          if (debug) {
            std::ostringstream os;
            os << "Sentence " << si << " IPA: '";
            for (char32_t cp : phonemes) {
              os << Utf32ToUtf8(std::u32string(1, cp));
            }
            os << "' -> IDs: [";
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
          // Wrap with PAD
          ids.insert(ids.begin(), 0);
          ids.push_back(0);
          token_ids.emplace_back(std::move(ids));
          ++si;
        }
      }
    } else {
      // Lexicon path: split text, tokenize each sentence, wrap with PAD
      std::string normalized = ReplacePunctuations(text);

      auto sentences = SplitByAllPunctuation(normalized);
      if (sentences.empty()) {
        SHERPA_ONNX_LOGE("No sentences after splitting");
        return {};
      }

      int32_t min_words = gen_config.GetExtraInt("min_words_in_sentence", 5);
      int32_t max_words = gen_config.GetExtraInt("max_words_in_sentence", 20);

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
        auto ids = TokenizeSentence(sentences[si], meta_data, debug, si);

        // Add trailing punctuation if not already present
        std::string trimmed = Trim(sentences[si]);
        if (!trimmed.empty()) {
          char last = trimmed.back();
          std::string punct(1, last);
          if (IsPunctuation(punct) && !ids.empty() &&
              !ids.back().tokens.empty()) {
            auto it = token_str2id_.find(punct);
            if (it != token_str2id_.end() &&
                ids.back().tokens.back() != it->second) {
              ids.back().tokens.push_back(it->second);
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

      auto audio = Process(batch_x, sid, speed, gen_config.silence_scale);
      ans.sample_rate = audio.sample_rate;
      ans.samples.insert(ans.samples.end(), audio.samples.begin(),
                         audio.samples.end());
      if (callback) {
        should_continue = callback(audio.samples.data(), audio.samples.size(),
                                   (b + 1) * 1.0 / num_batches);
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
      const std::string &text, const OfflineTtsKokoroModelMetaData &meta_data,
      bool debug, int32_t sentence_index = 0) const {
    if (!lexicon_str_.empty()) {
      return TokenizeFromLexiconStr(text, meta_data, debug, sentence_index);
    }

    SHERPA_ONNX_LOGE(
        "No lexicon available. "
        "Please provide --kokoro-lexicon.");
    return {};
  }

  std::vector<TokenIDs> TokenizeFromLexiconStr(
      const std::string &text, const OfflineTtsKokoroModelMetaData &meta_data,
      bool debug, int32_t sentence_index = 0) const {
    std::vector<TokenIDs> result;

    std::string normalized = ReplacePunctuations(text);

    if (debug) {
#if __OHOS__
      SHERPA_ONNX_LOGE("TokenizeFromLexiconStr: text='%{public}s'",
                       normalized.c_str());
#else
      SHERPA_ONNX_LOGE("TokenizeFromLexiconStr: text='%s'", normalized.c_str());
#endif
    }

    {
      std::vector<std::string> words;
      std::u32string u32 = Utf8ToUtf32(normalized);
      size_t current_word_start = 0;
      bool in_word = false;

      auto CjkPunctToAscii = [](char32_t c) -> const char * {
        if (c == U'，') return ",";
        if (c == U'、') return ",";
        if (c == U'。') return ".";
        if (c == U'！') return "!";
        if (c == U'？') return "?";
        if (c == U'；') return ";";
        if (c == U'：') return ",";
        if (c == U'…') return "...";
        return nullptr;
      };

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
          if (in_word) {
            words.push_back(Utf32ToUtf8(
                u32.substr(current_word_start, ci - current_word_start)));
            in_word = false;
          }
          words.push_back(Utf32ToUtf8(u32.substr(ci, 1)));
        } else if (CjkPunctToAscii(c)) {
          if (in_word) {
            words.push_back(Utf32ToUtf8(
                u32.substr(current_word_start, ci - current_word_start)));
            in_word = false;
          }
          words.push_back(CjkPunctToAscii(c));
        } else if (IsCJK(c)) {
          if (in_word) {
            words.push_back(Utf32ToUtf8(
                u32.substr(current_word_start, ci - current_word_start)));
            in_word = false;
          }
          words.push_back(Utf32ToUtf8(u32.substr(ci, 1)));
        } else {
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
        // Skip space tokens
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
            auto ids = ConvertPhonemeStringsToIds(it->second);
            sentence_ids.insert(sentence_ids.end(), ids.begin(), ids.end());
            i += len;
            found = true;
            break;
          }
        }

        if (!found) {
          if (ContainsCJK(words[i])) {
            std::vector<std::string> chars = SplitUtf8(words[i]);
            for (const auto &ch : chars) {
              std::string ch_lower = ToLowerCase(ch);
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
                std::string part_lower = ToLowerCase(part);
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
          std::ostringstream os;
          os << "Sentence " << sentence_index << " tokens: [";
          for (size_t j = 0; j < sentence_ids.size(); ++j) {
            if (j > 0) os << ", ";
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

        // Wrap with PAD (kokoro requires PAD at start and end)
        std::vector<int64_t> wrapped;
        wrapped.push_back(0);  // PAD
        wrapped.insert(wrapped.end(), sentence_ids.begin(), sentence_ids.end());
        wrapped.push_back(0);  // PAD
        result.emplace_back(std::move(wrapped));
      }
    }

    return result;
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

  static std::string ReplacePunctuations(const std::string &s) {
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
    static const std::string puncts = ",.;:!?";
    return s.size() == 1 && puncts.find(s[0]) != std::string::npos;
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

    Ort::Value audio = model_->Run(std::move(x_tensor), sid, speed);

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
  std::unordered_map<char32_t, int32_t> token2id_;         // for IPA codepoints
  std::unordered_map<std::string, int32_t> token_str2id_;  // for string tokens
  std::unordered_map<std::string, std::vector<std::string>> lexicon_str_;
  int32_t max_lexicon_phrase_len_ = 1;
};

}  // namespace sherpa_onnx
#endif  // SHERPA_ONNX_CSRC_OFFLINE_TTS_KOKORO_IMPL_H_
