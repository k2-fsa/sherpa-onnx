// sherpa-onnx/csrc/qwen-asr-tokenizer.cc
//
// Copyright (c)  2026  zengyw

#include "sherpa-onnx/csrc/qwen-asr-tokenizer.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
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

namespace {

using json = nlohmann::json;

std::string ToString(const std::vector<char> &data) {
  if (data.empty()) {
    return "";
  }

  return std::string(data.data(), data.size());
}

std::string ReadTextFile(const std::string &path) {
  return ToString(ReadFile(path));
}

#if __ANDROID_API__ >= 9
std::string ReadTextFile(AAssetManager *mgr, const std::string &path) {
  return ToString(ReadFile(mgr, path));
}
#endif

#if __OHOS__
std::string ReadTextFile(NativeResourceManager *mgr, const std::string &path) {
  return ToString(ReadFile(mgr, path));
}
#endif

std::string Utf8Encode(uint32_t cp) {
  std::string out;
  if (cp <= 0x7F) {
    out.push_back(static_cast<char>(cp));
  } else if (cp <= 0x7FF) {
    out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else if (cp <= 0xFFFF) {
    out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else {
    out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  }

  return out;
}

std::once_flag g_byte_to_unicode_once;
std::array<std::string, 256> g_byte_to_unicode;
std::unordered_map<std::string, uint8_t> g_unicode_to_byte;
std::once_flag g_oov_log_once;

void InitByteToUnicode() {
  std::array<bool, 256> used{};

  std::vector<int32_t> bytes;
  bytes.reserve(256);
  std::vector<uint32_t> code_points;
  code_points.reserve(256);

  for (int32_t b = 33; b <= 126; ++b) {
    bytes.push_back(b);
    used[static_cast<size_t>(b)] = true;
  }

  for (int32_t b = 161; b <= 172; ++b) {
    bytes.push_back(b);
    used[static_cast<size_t>(b)] = true;
  }

  for (int32_t b = 174; b <= 255; ++b) {
    bytes.push_back(b);
    used[static_cast<size_t>(b)] = true;
  }

  for (int32_t b : bytes) {
    code_points.push_back(static_cast<uint32_t>(b));
  }

  uint32_t n = 0;
  for (int32_t b = 0; b != 256; ++b) {
    if (!used[static_cast<size_t>(b)]) {
      bytes.push_back(b);
      code_points.push_back(256u + n);
      ++n;
    }
  }

  for (size_t i = 0; i != bytes.size(); ++i) {
    g_byte_to_unicode[static_cast<size_t>(bytes[i])] = Utf8Encode(code_points[i]);
  }

  g_unicode_to_byte.clear();
  g_unicode_to_byte.reserve(256);
  for (int32_t b = 0; b != 256; ++b) {
    g_unicode_to_byte[g_byte_to_unicode[static_cast<size_t>(b)]] =
        static_cast<uint8_t>(b);
  }
}

const std::array<std::string, 256> &ByteToUnicode() {
  std::call_once(g_byte_to_unicode_once, InitByteToUnicode);
  return g_byte_to_unicode;
}

const std::unordered_map<std::string, uint8_t> &UnicodeToByte() {
  std::call_once(g_byte_to_unicode_once, InitByteToUnicode);
  return g_unicode_to_byte;
}

bool Utf8Next(const std::string &s, size_t *i, uint32_t *cp, size_t *num_bytes) {
  if (*i >= s.size()) {
    return false;
  }

  size_t pos = *i;
  auto c = static_cast<unsigned char>(s[pos]);

  if (c < 0x80) {
    *cp = c;
    *num_bytes = 1;
  } else if ((c >> 5) == 0x6) {
    if (pos + 1 >= s.size()) {
      return false;
    }

    *cp = (c & 0x1F) << 6;
    *cp |= (static_cast<unsigned char>(s[pos + 1]) & 0x3F);
    *num_bytes = 2;
  } else if ((c >> 4) == 0xE) {
    if (pos + 2 >= s.size()) {
      return false;
    }

    *cp = (c & 0x0F) << 12;
    *cp |= (static_cast<unsigned char>(s[pos + 1]) & 0x3F) << 6;
    *cp |= (static_cast<unsigned char>(s[pos + 2]) & 0x3F);
    *num_bytes = 3;
  } else if ((c >> 3) == 0x1E) {
    if (pos + 3 >= s.size()) {
      return false;
    }

    *cp = (c & 0x07) << 18;
    *cp |= (static_cast<unsigned char>(s[pos + 1]) & 0x3F) << 12;
    *cp |= (static_cast<unsigned char>(s[pos + 2]) & 0x3F) << 6;
    *cp |= (static_cast<unsigned char>(s[pos + 3]) & 0x3F);
    *num_bytes = 4;
  } else {
    return false;
  }

  *i = pos + *num_bytes;
  return true;
}

std::vector<std::string> SplitUtf8ToChars(const std::string &s) {
  std::vector<std::string> ans;
  ans.reserve(s.size());

  size_t i = 0;
  while (i < s.size()) {
    uint32_t cp = 0;
    size_t num_bytes = 0;
    if (Utf8Next(s, &i, &cp, &num_bytes)) {
      ans.push_back(s.substr(i - num_bytes, num_bytes));
    } else {
      ans.push_back(s.substr(i, 1));
      ++i;
    }
  }

  return ans;
}

bool IsLetter(uint32_t cp) {
  if (cp >= 'a' && cp <= 'z') return true;
  if (cp >= 'A' && cp <= 'Z') return true;
  if (cp >= 0x4E00 && cp <= 0x9FFF) return true;
  if (cp >= 0x3400 && cp <= 0x4DBF) return true;
  if (cp >= 0x3040 && cp <= 0x30FF) return true;
  if (cp >= 0xAC00 && cp <= 0xD7AF) return true;
  return false;
}

std::vector<std::string> SplitByQwen3Pattern(const std::string &text) {
  std::vector<std::string> ans;
  ans.reserve(text.size() / 2 + 1);

  size_t start = 0;
  size_t i = 0;
  while (i < text.size()) {
    uint32_t cp = 0;
    size_t num_bytes = 0;
    if (Utf8Next(text, &i, &cp, &num_bytes)) {
      if (IsLetter(cp)) {
        continue;
      }

      if (i - num_bytes > start) {
        ans.push_back(text.substr(start, i - num_bytes - start));
      }
      ans.push_back(text.substr(i - num_bytes, num_bytes));
      start = i;
      continue;
    }

    if (i > start) {
      ans.push_back(text.substr(start, i - start));
    }
    ans.push_back(text.substr(i, 1));
    ++i;
    start = i;
  }

  if (i > start) {
    ans.push_back(text.substr(start, i - start));
  }

  return ans;
}

std::string MakeMergeKey(const std::string &left, const std::string &right) {
  std::string key;
  key.reserve(left.size() + right.size() + 1);
  key.append(left);
  key.push_back('\t');
  key.append(right);
  return key;
}

std::string ByteLevelEncode(const std::string &token,
                            const std::array<std::string, 256> &byte_to_unicode) {
  std::string ans;
  ans.reserve(token.size() * 2);
  for (unsigned char c : token) {
    ans.append(byte_to_unicode[c]);
  }
  return ans;
}

bool IsSpecialToken(const std::string &token) {
  if (token.size() < 5) {
    return false;
  }

  if (token.rfind("<|", 0) != 0) {
    return false;
  }

  return token.compare(token.size() - 2, 2, "|>") == 0;
}

bool ParseInt32String(const std::string &s, int32_t *value) {
  if (!value || s.empty()) {
    return false;
  }

  int64_t v = 0;
  for (char c : s) {
    if (c < '0' || c > '9') {
      return false;
    }

    v = v * 10 + (c - '0');
    if (v > std::numeric_limits<int32_t>::max()) {
      return false;
    }
  }

  *value = static_cast<int32_t>(v);
  return true;
}

bool ParseVocab(const std::string &content,
                std::unordered_map<std::string, int32_t> *token2id,
                std::vector<std::string> *id2token) {
  if (!token2id || !id2token) {
    return false;
  }

  token2id->clear();
  id2token->clear();

  json j = json::parse(content, nullptr, false);
  if (j.is_discarded() || !j.is_object()) {
    return false;
  }

  for (const auto &item : j.items()) {
    if (!item.value().is_number_integer()) {
      continue;
    }

    int64_t id64 = -1;
    try {
      id64 = item.value().get<int64_t>();
    } catch (...) {
      continue;
    }

    if (id64 < 0 || id64 > std::numeric_limits<int32_t>::max()) {
      continue;
    }

    int32_t id = static_cast<int32_t>(id64);
    (*token2id)[item.key()] = id;
    if (static_cast<size_t>(id) >= id2token->size()) {
      id2token->resize(static_cast<size_t>(id) + 1);
    }
    (*id2token)[static_cast<size_t>(id)] = item.key();
  }

  return !token2id->empty();
}

void ParseAddedTokens(const std::string &content,
                      std::unordered_map<std::string, int32_t> *token2id,
                      std::vector<std::string> *id2token) {
  if (!token2id || !id2token || content.empty()) {
    return;
  }

  json j = json::parse(content, nullptr, false);
  if (j.is_discarded() || !j.is_object()) {
    return;
  }

  auto it = j.find("added_tokens_decoder");
  if (it == j.end() || !it->is_object()) {
    return;
  }

  for (const auto &item : it->items()) {
    if (!item.value().is_object()) {
      continue;
    }

    auto content_it = item.value().find("content");
    if (content_it == item.value().end() || !content_it->is_string()) {
      continue;
    }

    int32_t id = -1;
    auto id_it = item.value().find("id");
    if (id_it != item.value().end() && id_it->is_number_integer()) {
      int64_t id64 = -1;
      try {
        id64 = id_it->get<int64_t>();
      } catch (...) {
        id64 = -1;
      }
      if (id64 >= 0 && id64 <= std::numeric_limits<int32_t>::max()) {
        id = static_cast<int32_t>(id64);
      }
    }

    if (id < 0 && !ParseInt32String(item.key(), &id)) {
      continue;
    }

    const std::string token = content_it->get<std::string>();
    if (token.empty()) {
      continue;
    }

    (*token2id)[token] = id;
    if (static_cast<size_t>(id) >= id2token->size()) {
      id2token->resize(static_cast<size_t>(id) + 1);
    }
    (*id2token)[static_cast<size_t>(id)] = token;
  }
}

bool ParseMerges(const std::string &content,
                 std::unordered_map<std::string, int32_t> *merges_rank) {
  if (!merges_rank) {
    return false;
  }

  merges_rank->clear();
  std::istringstream is(content);
  std::string line;
  int32_t rank = 0;

  while (std::getline(is, line)) {
    if (line.empty() || line[0] == '#') {
      continue;
    }

    std::istringstream ls(line);
    std::string left;
    std::string right;
    if (!(ls >> left >> right)) {
      continue;
    }

    (*merges_rank)[MakeMergeKey(left, right)] = rank++;
  }

  return true;
}

std::vector<std::string> BpeEncode(
    const std::string &word,
    const std::unordered_map<std::string, int32_t> &merges_rank,
    std::unordered_map<std::string, std::vector<std::string>> *cache,
    std::mutex *cache_mutex) {
  if (!word.empty() && cache && cache_mutex) {
    std::lock_guard<std::mutex> lock(*cache_mutex);
    auto it = cache->find(word);
    if (it != cache->end()) {
      return it->second;
    }
  }

  std::vector<std::string> symbols = SplitUtf8ToChars(word);
  if (symbols.size() <= 1) {
    if (cache && cache_mutex) {
      std::lock_guard<std::mutex> lock(*cache_mutex);
      (*cache)[word] = symbols;
    }
    return symbols;
  }

  while (symbols.size() > 1) {
    int32_t best_rank = std::numeric_limits<int32_t>::max();
    int32_t best_pos = -1;

    for (int32_t i = 0; i + 1 < static_cast<int32_t>(symbols.size()); ++i) {
      auto it = merges_rank.find(MakeMergeKey(symbols[i], symbols[i + 1]));
      if (it != merges_rank.end() && it->second < best_rank) {
        best_rank = it->second;
        best_pos = i;
      }
    }

    if (best_pos < 0) {
      break;
    }

    const std::string left = symbols[best_pos];
    const std::string right = symbols[best_pos + 1];
    std::vector<std::string> next;
    next.reserve(symbols.size());
    for (size_t i = 0; i < symbols.size();) {
      if (i + 1 < symbols.size() && symbols[i] == left &&
          symbols[i + 1] == right) {
        next.push_back(left + right);
        i += 2;
      } else {
        next.push_back(symbols[i]);
        ++i;
      }
    }
    symbols.swap(next);
  }

  if (cache && cache_mutex) {
    std::lock_guard<std::mutex> lock(*cache_mutex);
    (*cache)[word] = symbols;
  }

  return symbols;
}

void BuildSpecialTokens(
    const std::unordered_map<std::string, int32_t> &token2id,
    std::vector<std::pair<std::string, int32_t>> *special_tokens) {
  if (!special_tokens) {
    return;
  }

  special_tokens->clear();
  special_tokens->reserve(token2id.size());

  for (const auto &kv : token2id) {
    if (IsSpecialToken(kv.first)) {
      special_tokens->push_back(kv);
    }
  }

  std::sort(special_tokens->begin(), special_tokens->end(),
            [](const std::pair<std::string, int32_t> &a,
               const std::pair<std::string, int32_t> &b) {
              return a.first.size() > b.first.size();
            });
}

std::string DecodeBytes(const std::string &text) {
  const auto &unicode_to_byte = UnicodeToByte();

  std::string ans;
  ans.reserve(text.size());

  size_t i = 0;
  while (i < text.size()) {
    size_t pos = i;
    uint32_t cp = 0;
    size_t num_bytes = 0;
    if (Utf8Next(text, &i, &cp, &num_bytes)) {
      std::string piece = text.substr(pos, num_bytes);
      auto it = unicode_to_byte.find(piece);
      if (it != unicode_to_byte.end()) {
        ans.push_back(static_cast<char>(it->second));
      } else {
        ans.append(piece);
      }
      continue;
    }

    ans.push_back(text[i]);
    ++i;
  }

  return ans;
}

void AppendEncodedPieceIds(
    const std::string &encoded,
    const std::unordered_map<std::string, int32_t> &token2id,
    const std::unordered_map<std::string, int32_t> &merges_rank,
    std::unordered_map<std::string, std::vector<std::string>> *bpe_cache,
    std::mutex *bpe_cache_mutex, int64_t unk_token_id,
    std::vector<int64_t> *ids) {
  auto bpe_tokens = BpeEncode(encoded, merges_rank, bpe_cache, bpe_cache_mutex);

  size_t old_size = ids->size();
  bool has_missing = false;
  for (const auto &bpe_token : bpe_tokens) {
    auto it = token2id.find(bpe_token);
    if (it != token2id.end()) {
      ids->push_back(it->second);
    } else {
      has_missing = true;
      break;
    }
  }

  if (!has_missing) {
    return;
  }

  ids->resize(old_size);
  auto chars = SplitUtf8ToChars(encoded);
  for (const auto &c : chars) {
    auto it = token2id.find(c);
    if (it != token2id.end()) {
      ids->push_back(it->second);
      continue;
    }

    if (unk_token_id >= 0) {
      ids->push_back(unk_token_id);
      continue;
    }

    std::call_once(g_oov_log_once, []() {
      SHERPA_ONNX_LOGE(
          "qwen-asr-tokenizer: encountered OOV piece without <unk>; "
          "unmatched pieces will be dropped");
    });
  }
}

}  // namespace

QwenAsrTokenizer::QwenAsrTokenizer(const std::string &tokenizer_dir) {
  Init(tokenizer_dir);
}

#if __ANDROID_API__ >= 9
QwenAsrTokenizer::QwenAsrTokenizer(AAssetManager *mgr,
                                   const std::string &tokenizer_dir) {
  Init(mgr, tokenizer_dir);
}
#endif

#if __OHOS__
QwenAsrTokenizer::QwenAsrTokenizer(NativeResourceManager *mgr,
                                   const std::string &tokenizer_dir) {
  Init(mgr, tokenizer_dir);
}
#endif

void QwenAsrTokenizer::InitFromContents(const std::string &vocab_content,
                                        const std::string &merges_content,
                                        const std::string &config_content,
                                        const std::string &tokenizer_dir) {
  if (vocab_content.empty()) {
    SHERPA_ONNX_LOGE("Failed to read vocab.json from: %s", tokenizer_dir.c_str());
    SHERPA_ONNX_EXIT(-1);
  }

  token2id_.clear();
  id2token_.clear();
  special_tokens_.clear();
  merges_rank_.clear();
  byte_to_unicode_ = ByteToUnicode();

  if (!ParseVocab(vocab_content, &token2id_, &id2token_)) {
    SHERPA_ONNX_LOGE("Failed to parse vocab.json from: %s", tokenizer_dir.c_str());
    SHERPA_ONNX_EXIT(-1);
  }

  if (!merges_content.empty() && !ParseMerges(merges_content, &merges_rank_)) {
    SHERPA_ONNX_LOGE("Failed to parse merges.txt from: %s", tokenizer_dir.c_str());
    SHERPA_ONNX_EXIT(-1);
  }

  ParseAddedTokens(config_content, &token2id_, &id2token_);
  BuildSpecialTokens(token2id_, &special_tokens_);

  {
    std::lock_guard<std::mutex> lock(bpe_cache_mutex_);
    bpe_cache_.clear();
  }

  eos_token_id_ = -1;
  pad_token_id_ = -1;
  im_end_token_id_ = -1;
  unk_token_id_ = -1;

  auto it = token2id_.find("<|im_end|>");
  if (it != token2id_.end()) {
    eos_token_id_ = it->second;
    im_end_token_id_ = it->second;
  }

  it = token2id_.find("<|padding|>");
  if (it != token2id_.end()) {
    pad_token_id_ = it->second;
  }

  if (pad_token_id_ < 0) {
    it = token2id_.find("<|endoftext|>");
    if (it != token2id_.end()) {
      pad_token_id_ = it->second;
    }
  }

  if (pad_token_id_ < 0) {
    pad_token_id_ = eos_token_id_;
  }

  it = token2id_.find("<unk>");
  if (it != token2id_.end()) {
    unk_token_id_ = it->second;
  } else {
    it = token2id_.find("<|unk|>");
    if (it != token2id_.end()) {
      unk_token_id_ = it->second;
    }
  }
}

void QwenAsrTokenizer::Init(const std::string &tokenizer_dir) {
  const std::string vocab_path = tokenizer_dir + "/vocab.json";
  const std::string merges_path = tokenizer_dir + "/merges.txt";
  const std::string config_path = tokenizer_dir + "/tokenizer_config.json";

  InitFromContents(ReadTextFile(vocab_path), ReadTextFile(merges_path),
                   ReadTextFile(config_path), tokenizer_dir);
}

#if __ANDROID_API__ >= 9
void QwenAsrTokenizer::Init(AAssetManager *mgr,
                            const std::string &tokenizer_dir) {
  const std::string vocab_path = tokenizer_dir + "/vocab.json";
  const std::string merges_path = tokenizer_dir + "/merges.txt";
  const std::string config_path = tokenizer_dir + "/tokenizer_config.json";

  InitFromContents(ReadTextFile(mgr, vocab_path), ReadTextFile(mgr, merges_path),
                   ReadTextFile(mgr, config_path), tokenizer_dir);
}
#endif

#if __OHOS__
void QwenAsrTokenizer::Init(NativeResourceManager *mgr,
                            const std::string &tokenizer_dir) {
  const std::string vocab_path = tokenizer_dir + "/vocab.json";
  const std::string merges_path = tokenizer_dir + "/merges.txt";
  const std::string config_path = tokenizer_dir + "/tokenizer_config.json";

  InitFromContents(ReadTextFile(mgr, vocab_path), ReadTextFile(mgr, merges_path),
                   ReadTextFile(mgr, config_path), tokenizer_dir);
}
#endif

std::vector<int64_t> QwenAsrTokenizer::Encode(const std::string &text) {
  std::vector<int64_t> ans;

  size_t pos = 0;
  while (pos < text.size()) {
    bool matched_special = false;
    for (const auto &token : special_tokens_) {
      if (pos + token.first.size() <= text.size() &&
          text.compare(pos, token.first.size(), token.first) == 0) {
        ans.push_back(token.second);
        pos += token.first.size();
        matched_special = true;
        break;
      }
    }

    if (matched_special) {
      continue;
    }

    size_t next_special_pos = text.size();
    for (const auto &token : special_tokens_) {
      size_t p = text.find(token.first, pos);
      if (p != std::string::npos && p < next_special_pos) {
        next_special_pos = p;
      }
    }

    if (next_special_pos > pos) {
      std::string chunk = text.substr(pos, next_special_pos - pos);
      auto pieces = SplitByQwen3Pattern(chunk);
      for (const auto &piece : pieces) {
        std::string encoded = ByteLevelEncode(piece, byte_to_unicode_);
        AppendEncodedPieceIds(encoded, token2id_, merges_rank_, &bpe_cache_,
                              &bpe_cache_mutex_, unk_token_id_, &ans);
      }
    }

    pos = next_special_pos;
  }

  return ans;
}

std::string QwenAsrTokenizer::Decode(const std::vector<int64_t> &token_ids) {
  std::string ans;
  std::string buffer;

  for (int64_t id : token_ids) {
    if (id < 0 || static_cast<size_t>(id) >= id2token_.size()) {
      continue;
    }

    const std::string &token = id2token_[static_cast<size_t>(id)];
    if (IsSpecialToken(token)) {
      if (!buffer.empty()) {
        ans.append(DecodeBytes(buffer));
        buffer.clear();
      }
      ans.append(token);
    } else {
      buffer.append(token);
    }
  }

  if (!buffer.empty()) {
    ans.append(DecodeBytes(buffer));
  }

  return ans;
}

std::string QwenAsrTokenizer::GetTokenStringStreaming(int64_t token_id,
                                                      std::string *state) const {
  if (token_id < 0 || static_cast<size_t>(token_id) >= id2token_.size()) {
    return "";
  }

  const std::string &token = id2token_[static_cast<size_t>(token_id)];
  if (IsSpecialToken(token)) {
    if (state && !state->empty()) {
      std::string out = DecodeBytes(*state);
      state->clear();
      out.append(token);
      return out;
    }
    return token;
  }

  std::string bytes = DecodeBytes(token);
  if (!state) {
    return bytes;
  }

  state->append(bytes);

  size_t i = 0;
  size_t last_good = 0;
  while (i < state->size()) {
    uint32_t cp = 0;
    size_t num_bytes = 0;
    if (!Utf8Next(*state, &i, &cp, &num_bytes)) {
      break;
    }
    last_good = i;
  }

  if (last_good == 0) {
    return "";
  }

  std::string ans = state->substr(0, last_good);
  state->erase(0, last_good);
  return ans;
}

}  // namespace sherpa_onnx
