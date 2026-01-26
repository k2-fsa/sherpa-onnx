// sherpa-onnx/csrc/sentence-piece-tokenizer.cc
//
// Copyright (c)  2026  Xiaomi Corporation

#include "sherpa-onnx/csrc/sentence-piece-tokenizer.h"

#include <cstdio>
#include <fstream>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#if __OHOS__
#include "rawfile/raw_file_manager.h"
#endif

#include "nlohmann/json.hpp"
#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {

using json = nlohmann::json;
static constexpr float kNegInf = -1e30f;

static json LoadJson(const std::string &filename) {
  if (filename.empty()) {
    SHERPA_ONNX_LOGE("Empty json filename");
    SHERPA_ONNX_EXIT(-1);
  }
  AssertFileExists(filename);

  std::ifstream is(filename);
  json j;
  is >> j;
  return j;
}

static json LoadJson(const std::vector<char> &buf) {
  if (buf.empty()) {
    SHERPA_ONNX_LOGE("Empty json buffer");
    SHERPA_ONNX_EXIT(-1);
  }
  return json::parse(buf.begin(), buf.end());
}

class SentencePieceTokenizer::Impl {
 public:
  Impl(const std::string &vocab_json, const std::string &token_scores_json) {
    Init(LoadJson(vocab_json), LoadJson(token_scores_json));
  }

  template <typename Manager>
  Impl(Manager *mgr, const std::string &vocab_json,
       const std::string &token_scores_json) {
    Init(LoadJson(ReadFile(mgr, vocab_json)),
         LoadJson(ReadFile(mgr, token_scores_json)));
  }

  std::vector<int32_t> EncodeIds(const std::string &text) const {
    std::vector<int32_t> ids;
    EncodeInternal(text, &ids, nullptr);
    return ids;
  }

  std::vector<std::string> EncodeTokens(const std::string &text) const {
    std::vector<std::string> tokens;
    EncodeInternal(text, nullptr, &tokens);
    return tokens;
  }

 private:
  void Init(const json &vocab, const json &scores) {
    InitVocabJson(vocab);
    InitTokenScores(scores);

    for (int i = 0; i < 256; ++i) {
      byte_token_id_[i] = -1;
      byte_token_score_[i] = kNegInf;
    }

    InitTrie();
  }

  void InitVocabJson(const std::string &filename) {
    InitVocabJson(LoadJson(filename));
  }

  void InitVocabJson(const std::vector<char> &buf) {
    InitVocabJson(LoadJson(buf));
  }

  void InitVocabJson(const json &j) {
    token2id_.reserve(j.size());
    id2token_.resize(j.size());

    for (const auto &item : j.items()) {
      token2id_[item.key()] = item.value();
      id2token_[item.value()] = item.key();
    }
  }

  void InitTokenScores(const std::string &filename) {
    InitTokenScores(LoadJson(filename));
  }

  void InitTokenScores(const std::vector<char> &buf) {
    InitTokenScores(LoadJson(buf));
  }

  void InitTokenScores(const json &j) {
    token2score_.reserve(j.size());

    for (const auto &item : j.items()) {
      token2score_[item.key()] = item.value();
    }
  }

  void InitTrie() {
    trie_.reserve(token2id_.size() * 2);
    trie_.push_back(TrieNode());  // root

    for (const auto &kv : token2id_) {
      const std::string &tok = kv.first;
      int32_t id = kv.second;

      int32_t node = 0;
      for (unsigned char c : tok) {
        auto it = trie_[node].next.find(c);
        if (it == trie_[node].next.end()) {
          int32_t new_node = trie_.size();
          trie_[node].next[c] = new_node;
          trie_.push_back(TrieNode());
          node = new_node;
        } else {
          node = it->second;
        }
      }

      trie_[node].token_id = id;
      trie_[node].score = token2score_[tok];
    }

    // -------------------------
    // Byte fallback
    // -------------------------
    for (int32_t i = 0; i < 256; ++i) {
      char buf[8];
      std::snprintf(buf, sizeof(buf), "<0x%02X>", i);
      std::string tok(buf);

      auto it = token2id_.find(tok);
      if (it == token2id_.end()) {
        SHERPA_ONNX_LOGE("Missing byte token: '%s'", tok.c_str());
        continue;
      }

      byte_token_id_[i] = it->second;
      byte_token_score_[i] = token2score_[tok];
    }
  }

  void EncodeInternal(const std::string &input, std::vector<int32_t> *ids,
                      std::vector<std::string> *tokens) const {
    // SentencePiece whitespace handling
    std::string text;
    text.reserve(input.size() + 8);

    for (char c : input) {
      if (c == ' ')
        text.append("\xE2\x96\x81");  // ▁
      else
        text.push_back(c);
    }

    if (text.rfind("\xE2\x96\x81", 0) == std::string::npos) {
      text.insert(0, "\xE2\x96\x81");
    }

    const int32_t n = static_cast<int32_t>(text.size());
    std::vector<float> dp(n + 1, kNegInf);
    std::vector<int32_t> back(n + 1, -1);
    std::vector<int32_t> back_id(n + 1, -1);

    dp[n] = 0.0f;

    // DP
    for (int32_t i = n - 1; i >= 0; --i) {
      int32_t node = 0;
      for (int32_t j = i; j < n; ++j) {
        unsigned char c = static_cast<unsigned char>(text[j]);
        auto it = trie_[node].next.find(c);
        if (it == trie_[node].next.end()) break;
        node = it->second;

        if (trie_[node].token_id >= 0) {
          float score = trie_[node].score + dp[j + 1];
          if (score > dp[i]) {
            dp[i] = score;
            back[i] = j + 1;
            back_id[i] = trie_[node].token_id;
          }
        }
      }

      // byte fallback
      if (back[i] < 0) {
        unsigned char b = static_cast<unsigned char>(text[i]);
        dp[i] = byte_token_score_[b] + dp[i + 1];
        back[i] = i + 1;
        back_id[i] = byte_token_id_[b];
      }
    }

    // reconstruct
    for (int32_t i = 0; i < n;) {
      int32_t j = back[i];
      int32_t id = back_id[i];
      if (j <= i || id < 0) break;

      if (ids != nullptr) {
        ids->push_back(id);
      }

      if (tokens != nullptr) {
        tokens->push_back(id2token_[id]);
      }

      i = j;
    }
  }

 private:
  struct TrieNode {
    std::unordered_map<unsigned char, int32_t> next;
    int32_t token_id = -1;
    float score = 0.0f;
  };

  std::vector<TrieNode> trie_;  // immutable after build
  std::vector<std::string> id2token_;
  std::unordered_map<std::string, int32_t> token2id_;
  std::unordered_map<std::string, float> token2score_;

  // <0xNN> byte fallback
  int32_t byte_token_id_[256];
  float byte_token_score_[256];
};

SentencePieceTokenizer::SentencePieceTokenizer(
    const std::string &vocab_json, const std::string &token_scores_json)
    : impl_(std::make_unique<Impl>(vocab_json, token_scores_json)) {}

template <typename Manager>
SentencePieceTokenizer::SentencePieceTokenizer(
    Manager *mgr, const std::string &vocab_json,
    const std::string &token_scores_json)
    : impl_(std::make_unique<Impl>(mgr, vocab_json, token_scores_json)) {}

SentencePieceTokenizer::~SentencePieceTokenizer() = default;

std::vector<int32_t> SentencePieceTokenizer::EncodeIds(
    const std::string &text) const {
  return impl_->EncodeIds(text);
}

std::vector<std::string> SentencePieceTokenizer::EncodeTokens(
    const std::string &text) const {
  return impl_->EncodeTokens(text);
}

#if __ANDROID_API__ >= 9
template SentencePieceTokenizer::SentencePieceTokenizer(
    AAssetManager *mgr, const std::string &vocab_json,
    const std::string &token_scores_json);
#endif

#if __OHOS__
template SentencePieceTokenizer::SentencePieceTokenizer(
    NativeResourceManager *mgr, const std::string &vocab_json,
    const std::string &token_scores_json);
#endif

}  // namespace sherpa_onnx
