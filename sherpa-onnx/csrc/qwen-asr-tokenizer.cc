// sherpa-onnx/csrc/qwen-asr-tokenizer.cc
//
// Copyright (c)  2026  zengyw
//
// A simplified BPE tokenizer for Qwen3-ASR.
#include "sherpa-onnx/csrc/qwen-asr-tokenizer.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#include "sherpa-onnx/csrc/file-utils.h"
#endif

#if __OHOS__
#include "rawfile/raw_file_manager.h"
#include "sherpa-onnx/csrc/file-utils.h"
#endif

#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {

namespace {

static std::string ReadFileToString(const std::string &path) {
#if __ANDROID_API__ >= 9 || __OHOS__
  std::vector<char> data = ReadFile(path);
  if (data.empty()) {
    return "";
  }
  return std::string(data.data(), data.size());
#else
  std::ifstream ifs(path, std::ios::binary | std::ios::ate);
  if (!ifs) {
    return "";
  }
  size_t size = ifs.tellg();
  std::string content(size, '\0');
  ifs.seekg(0);
  ifs.read(content.data(), size);
  return content;
#endif
}

static inline void TrimInPlace(std::string *s) {
  if (!s) return;
  auto &x = *s;
  size_t b = x.find_first_not_of(" \t\r\n");
  if (b == std::string::npos) {
    x.clear();
    return;
  }
  size_t e = x.find_last_not_of(" \t\r\n");
  x = x.substr(b, e - b + 1);
}

static std::string Utf8Encode(uint32_t cp) {
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

static std::once_flag g_b2u_once;
static std::array<std::string, 256> g_byte_to_unicode;
static std::unordered_map<std::string, uint8_t> g_unicode_to_byte;

static void InitByteUnicodeMapsOnce() {
  bool in_bs[256];
  std::memset(in_bs, 0, sizeof(in_bs));

  std::vector<int> bs;
  bs.reserve(256);
  std::vector<uint32_t> cs;
  cs.reserve(256);

  for (int b = 33; b <= 126; ++b) {
    bs.push_back(b);
    in_bs[b] = true;
  }
  for (int b = 161; b <= 172; ++b) {
    bs.push_back(b);
    in_bs[b] = true;
  }
  for (int b = 174; b <= 255; ++b) {
    bs.push_back(b);
    in_bs[b] = true;
  }

  for (int b : bs) {
    cs.push_back(static_cast<uint32_t>(b));
  }

  uint32_t n = 0;
  for (int b = 0; b < 256; ++b) {
    if (!in_bs[b]) {
      bs.push_back(b);
      cs.push_back(256u + n);
      ++n;
    }
  }

  for (size_t i = 0; i < bs.size(); ++i) {
    g_byte_to_unicode[static_cast<size_t>(bs[i])] = Utf8Encode(cs[i]);
  }

  g_unicode_to_byte.clear();
  g_unicode_to_byte.reserve(256);
  for (int b = 0; b < 256; ++b) {
    g_unicode_to_byte[g_byte_to_unicode[static_cast<size_t>(b)]] =
        static_cast<uint8_t>(b);
  }
}

static inline const std::array<std::string, 256> &ByteToUnicode() {
  std::call_once(g_b2u_once, InitByteUnicodeMapsOnce);
  return g_byte_to_unicode;
}

static inline const std::unordered_map<std::string, uint8_t> &UnicodeToByte() {
  std::call_once(g_b2u_once, InitByteUnicodeMapsOnce);
  return g_unicode_to_byte;
}

// UTF-8 utilities (same as FunASRNanoTokenizer)
static bool Utf8Next(const std::string &s, size_t *i, uint32_t *cp, size_t *nb) {
  if (*i >= s.size()) return false;
  size_t pos = *i;
  unsigned char c = static_cast<unsigned char>(s[pos]);

  if (c < 0x80) {
    *cp = c;
    *nb = 1;
  } else if ((c >> 5) == 0x6) {
    if (pos + 1 >= s.size()) return false;
    *cp = (c & 0x1F) << 6;
    *cp |= (static_cast<unsigned char>(s[pos + 1]) & 0x3F);
    *nb = 2;
  } else if ((c >> 4) == 0xE) {
    if (pos + 2 >= s.size()) return false;
    *cp = (c & 0x0F) << 12;
    *cp |= (static_cast<unsigned char>(s[pos + 1]) & 0x3F) << 6;
    *cp |= (static_cast<unsigned char>(s[pos + 2]) & 0x3F);
    *nb = 3;
  } else if ((c >> 3) == 0x1E) {
    if (pos + 3 >= s.size()) return false;
    *cp = (c & 0x07) << 18;
    *cp |= (static_cast<unsigned char>(s[pos + 1]) & 0x3F) << 12;
    *cp |= (static_cast<unsigned char>(s[pos + 2]) & 0x3F) << 6;
    *cp |= (static_cast<unsigned char>(s[pos + 3]) & 0x3F);
    *nb = 4;
  } else {
    return false;
  }

  *i = pos + *nb;
  return true;
}

static std::vector<std::string> SplitUtf8ToChars(const std::string &s) {
  std::vector<std::string> result;
  size_t i = 0;
  while (i < s.size()) {
    uint32_t cp = 0;
    size_t nb = 0;
    if (Utf8Next(s, &i, &cp, &nb)) {
      result.push_back(s.substr(i - nb, nb));
    } else {
      result.push_back(s.substr(i, 1));
      ++i;
    }
  }
  return result;
}

static inline bool IsLetter(uint32_t cp) {
  if (cp >= 'a' && cp <= 'z') return true;
  if (cp >= 'A' && cp <= 'Z') return true;
  // CJK Unified Ideographs
  if (cp >= 0x4E00 && cp <= 0x9FFF) return true;
  // CJK Extension A
  if (cp >= 0x3400 && cp <= 0x4DBF) return true;
  // Hiragana/Katakana
  if (cp >= 0x3040 && cp <= 0x30FF) return true;
  // Hangul syllables
  if (cp >= 0xAC00 && cp <= 0xD7AF) return true;
  return false;
}

static std::vector<std::string> SplitByQwen3Pattern(const std::string &s) {
  std::vector<std::string> words;
  size_t start = 0;
  size_t i = 0;

  while (i < s.size()) {
    uint32_t cp = 0;
    size_t nb = 0;
    if (Utf8Next(s, &i, &cp, &nb)) {
      if (IsLetter(cp)) {
        // Continue current word
      } else {
        if (i - nb > start) {
          words.push_back(s.substr(start, i - nb - start));
        }
        words.push_back(s.substr(i - nb, nb));
        start = i;
      }
    } else {
      if (i > start) {
        words.push_back(s.substr(start, i - start));
      }
      words.push_back(s.substr(i, 1));
      ++i;
      start = i;
    }
  }

  if (i > start) {
    words.push_back(s.substr(start, i - start));
  }

  return words;
}

static std::string MakeMergeKey(const std::string &first,
                               const std::string &second) {
  std::string k;
  k.reserve(first.size() + second.size() + 1);
  k.append(first);
  k.push_back('\t');
  k.append(second);
  return k;
}

// Parse vocab.json - format: {"token": id, ...}
static bool ParseVocabJson(const std::string &content,
                           std::unordered_map<std::string, int32_t> *vocab,
                           std::vector<std::string> *id2vocab) {
  if (!vocab || !id2vocab) return false;
  vocab->clear();
  id2vocab->clear();

  size_t pos = 0;
  while (pos < content.size() && content[pos] != '{') {
    ++pos;
  }
  if (pos >= content.size()) return false;
  ++pos;

  while (pos < content.size()) {
    while (pos < content.size() &&
           (content[pos] == ' ' || content[pos] == '\t' ||
            content[pos] == '\n' || content[pos] == '\r' ||
            content[pos] == ',')) {
      ++pos;
    }

    if (pos >= content.size() || content[pos] != '"') break;

    ++pos;
    size_t start = pos;

    while (pos < content.size() && content[pos] != '"') {
      if (content[pos] == '\\' && pos + 1 < content.size()) {
        ++pos;
      }
      ++pos;
    }

    if (pos >= content.size()) break;

    std::string token = content.substr(start, pos - start);
    ++pos;

    while (pos < content.size() && content[pos] != ':') {
      ++pos;
    }
    if (pos >= content.size()) break;
    ++pos;

    while (pos < content.size() &&
           (content[pos] == ' ' || content[pos] == '\t' ||
            content[pos] == '\n')) {
      ++pos;
    }

    if (pos >= content.size() || content[pos] < '0' || content[pos] > '9')
      break;

    int32_t id = 0;
    while (pos < content.size() && content[pos] >= '0' && content[pos] <= '9') {
      id = id * 10 + (content[pos] - '0');
      ++pos;
    }

    if (!token.empty()) {
      (*vocab)[token] = id;
      if (static_cast<size_t>(id) >= id2vocab->size()) {
        id2vocab->resize(id + 1);
      }
      (*id2vocab)[id] = token;
    }
  }

  return true;
}


static void ParseAddedTokensDecoder(
    const std::string &content, std::unordered_map<std::string, int32_t> *token2id,
    std::vector<std::string> *id2token,
    std::unordered_set<std::string> *added_tokens) {
  if (!token2id || !id2token) return;

  size_t section_start = content.find("\"added_tokens_decoder\"");
  if (section_start == std::string::npos) return;

  size_t brace_pos = content.find('{', section_start);
  if (brace_pos == std::string::npos) return;
  ++brace_pos;

  int depth = 1;
  size_t section_end = brace_pos;
  for (size_t i = brace_pos + 1; i < content.size(); ++i) {
    if (content[i] == '{')
      ++depth;
    else if (content[i] == '}') {
      --depth;
      if (depth == 0) {
        section_end = i + 1;
        break;
      }
    }
  }

  std::string section;
  if (section_end > brace_pos) {
    section = content.substr(brace_pos, section_end - brace_pos);
  }
  if (section.empty()) return;

  size_t pos = 0;
  while (pos < section.size()) {
    size_t q1 = section.find('"', pos);
    if (q1 == std::string::npos) {
      break;
    }

    size_t q2 = section.find('"', q1 + 1);
    if (q2 == std::string::npos) break;

    std::string key = section.substr(q1 + 1, q2 - q1 - 1);

    bool is_number = !key.empty();
    for (char c : key) {
      if (c < '0' || c > '9') {
        is_number = false;
        break;
      }
    }

    if (is_number) {
      size_t entry_start = q2 + 1;
      int entry_depth = 0;
      size_t entry_end = entry_start;
      for (size_t i = entry_start; i < section.size(); ++i) {
        if (section[i] == '{')
          ++entry_depth;
        else if (section[i] == '}') {
          --entry_depth;
          if (entry_depth == 0) {
            entry_end = i + 1;
            break;
          }
        }
      }

      std::string field_key = "\"content\"";
      size_t field_pos = section.find(field_key, entry_start);
      if (field_pos != std::string::npos && field_pos < entry_end) {
        size_t colon = section.find(':', field_pos);
        if (colon != std::string::npos && colon < entry_end) {
          // Find the opening quote after colon
          size_t value_start = section.find('"', colon + 1);
          if (value_start != std::string::npos && value_start < entry_end) {
            ++value_start;
            size_t value_end = section.find('"', value_start);
            if (value_end != std::string::npos && value_end <= entry_end) {
              std::string token_value =
                  section.substr(value_start, value_end - value_start);

              if (!token_value.empty()) {
                try {
                  int32_t token_id = std::stoi(key);
                  (*token2id)[token_value] = token_id;
                  if (static_cast<size_t>(token_id) >= id2token->size()) {
                    id2token->resize(token_id + 1);
                  }
                  (*id2token)[token_id] = token_value;
                  if (added_tokens) {
                    added_tokens->insert(token_value);
                  }
                } catch (const std::exception &e) {
                }
              }
            }
          }
        }
      }

      pos = entry_end;
    } else {
      pos = q2 + 1;
    }
  }
}

// Parse merges.txt
static bool ParseMergesTxt(
    const std::string &content,
    std::unordered_map<std::string, int32_t> *merges_rank) {
  if (!merges_rank) return false;
  merges_rank->clear();

  size_t pos = 0;
  int32_t rank = 0;

  while (pos < content.size()) {
    while (pos < content.size() &&
           (content[pos] == ' ' || content[pos] == '\t' ||
            content[pos] == '\n' || content[pos] == '\r')) {
      ++pos;
    }

    if (pos >= content.size()) break;

    if (content[pos] == '#') {
      while (pos < content.size() && content[pos] != '\n') {
        ++pos;
      }
      continue;
    }

    size_t start = pos;
    while (pos < content.size() && content[pos] != ' ' && content[pos] != '\t' &&
           content[pos] != '\n' && content[pos] != '\r') {
      ++pos;
    }
    if (pos <= start) break;
    std::string first = content.substr(start, pos - start);

    while (pos < content.size() && (content[pos] == ' ' || content[pos] == '\t')) {
      ++pos;
    }
    if (pos >= content.size()) break;

    start = pos;
    while (pos < content.size() && content[pos] != ' ' && content[pos] != '\t' &&
           content[pos] != '\n' && content[pos] != '\r') {
      ++pos;
    }
    if (pos <= start) break;
    std::string second = content.substr(start, pos - start);

    std::string key = MakeMergeKey(first, second);
    (*merges_rank)[key] = rank;
    ++rank;
  }

  return true;
}

// BPE encode with caching
static std::vector<std::string> BpeEncodeWithCache(
    const std::string &word,
    const std::unordered_map<std::string, int32_t> &merges_rank,
    std::unordered_map<std::string, std::vector<std::string>> *cache) {
  if (!word.empty()) {
    auto it = cache->find(word);
    if (it != cache->end()) {
      return it->second;
    }
  }

  std::vector<std::string> symbols = SplitUtf8ToChars(word);
  if (symbols.empty()) {
    if (cache) (*cache)[word] = {};
    return {};
  }
  if (symbols.size() == 1) {
    if (cache) (*cache)[word] = symbols;
    return symbols;
  }

  while (symbols.size() > 1) {
    int32_t best_rank = std::numeric_limits<int32_t>::max();
    int32_t best_pos = -1;

    for (int32_t i = 0; i + 1 < static_cast<int32_t>(symbols.size()); ++i) {
      std::string key = MakeMergeKey(symbols[i], symbols[i + 1]);
      auto it2 = merges_rank.find(key);
      if (it2 != merges_rank.end()) {
        int32_t r = it2->second;
        if (r < best_rank) {
          best_rank = r;
          best_pos = i;
        }
      }
    }

    if (best_pos < 0) break;

    std::string a = symbols[best_pos];
    std::string b = symbols[best_pos + 1];

    std::vector<std::string> new_symbols;
    new_symbols.reserve(symbols.size());

    for (size_t i = 0; i < symbols.size();) {
      if (i + 1 < symbols.size() && symbols[i] == a && symbols[i + 1] == b) {
        new_symbols.push_back(a + b);
        i += 2;
      } else {
        new_symbols.push_back(symbols[i]);
        i += 1;
      }
    }

    symbols.swap(new_symbols);
  }

  if (cache) (*cache)[word] = symbols;
  return symbols;
}

static inline bool IsSpecialTokenString(const std::string &s) {
  if (s.size() < 5) return false;
  if (s.rfind("<|", 0) == 0 && s.size() >= 2 && s.substr(s.size() - 2) == "|>") {
    return true;
  }
  return false;
}

static std::string DecodeBytesToUtf8(const std::string &u) {
  const auto &u2b = UnicodeToByte();
  std::string bytes;
  bytes.reserve(u.size());

  size_t i = 0;
  while (i < u.size()) {
    uint32_t cp = 0;
    size_t nb = 0;
    size_t pos0 = i;
    if (Utf8Next(u, &i, &cp, &nb)) {
      std::string k = u.substr(pos0, nb);
      auto it = u2b.find(k);
      if (it != u2b.end()) {
        bytes.push_back(static_cast<char>(it->second));
      } else {
        bytes.append(k);
      }
    } else {
      bytes.push_back(u[i]);
      ++i;
    }
  }

  return bytes;
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

void QwenAsrTokenizer::Init(const std::string &tokenizer_dir) {
  std::string vocab_path = tokenizer_dir + "/vocab.json";
  std::string merges_path = tokenizer_dir + "/merges.txt";
  std::string config_path = tokenizer_dir + "/tokenizer_config.json";

  std::string vocab_content = ReadFileToString(vocab_path);
  std::string merges_content = ReadFileToString(merges_path);
  std::string config_content = ReadFileToString(config_path);

  if (vocab_content.empty()) {
    SHERPA_ONNX_LOGE("Failed to read vocab.json from: %s",
                     tokenizer_dir.c_str());
    SHERPA_ONNX_EXIT(-1);
  }

  const auto &b2u = ByteToUnicode();
  for (int i = 0; i < 256; ++i) {
    byte_to_unicode_[i] = b2u[static_cast<size_t>(i)];
  }

  // Parse vocab.json
  if (!ParseVocabJson(vocab_content, &token2id_, &id2token_)) {
    SHERPA_ONNX_LOGE("Failed to parse vocab.json");
    SHERPA_ONNX_EXIT(-1);
  }

  // Parse merges.txt
  if (!merges_content.empty()) {
    if (!ParseMergesTxt(merges_content, &merges_rank_)) {
      SHERPA_ONNX_LOGE("Failed to parse merges.txt");
      SHERPA_ONNX_EXIT(-1);
    }
  }

  // Parse added_tokens_decoder from tokenizer_config.json
  if (!config_content.empty()) {
    ParseAddedTokensDecoder(config_content, &token2id_, &id2token_, &added_tokens_);
  }

  // Find special token IDs
  eos_token_id_ = -1;
  pad_token_id_ = -1;
  im_end_token_id_ = -1;

  for (const auto &[token, id] : token2id_) {
    if (token == "<|im_end|>") {
      eos_token_id_ = id;
      im_end_token_id_ = id;
    } else if (token == "<|endoftext|>" || token == "<|padding|>") {
      if (pad_token_id_ < 0) pad_token_id_ = id;
    }
  }

  for (size_t i = 0; i < id2token_.size(); ++i) {
    const std::string &token = id2token_[i];
    if (token == "<|im_end|>") {
      eos_token_id_ = static_cast<int64_t>(i);
      im_end_token_id_ = static_cast<int64_t>(i);
    } else if (token == "<|endoftext|>" || token == "<|padding|>") {
      if (pad_token_id_ < 0) pad_token_id_ = static_cast<int64_t>(i);
    }
  }
}

#if __ANDROID_API__ >= 9
void QwenAsrTokenizer::Init(AAssetManager *mgr,
                            const std::string &tokenizer_dir) {
  Init(tokenizer_dir);
}
#endif

#if __OHOS__
void QwenAsrTokenizer::Init(NativeResourceManager *mgr,
                            const std::string &tokenizer_dir) {
  Init(tokenizer_dir);
}
#endif

std::vector<int64_t> QwenAsrTokenizer::Encode(const std::string &text) {
  std::vector<int64_t> result;

  // Build list of special tokens (from added_tokens_decoder)
  // These are Qwen-specific special tokens like <|im_start|>, <|im_end|>, etc.
  std::vector<std::pair<std::string, int32_t>> special_tokens;
  for (const auto &[token, id] : token2id_) {
    // Qwen special tokens: start with "<|" and end with "|>"
    if (token.size() >= 5 &&
        token.find("<|") == 0 &&
        token.find("|>") == token.size() - 2) {
      special_tokens.push_back({token, id});
    }
  }

  // Sort by length descending for longest match first
  std::sort(special_tokens.begin(), special_tokens.end(),
            [](const auto &a, const auto &b) {
              return a.first.size() > b.first.size();
            });

  // Process text: check for special tokens at each position
  size_t pos = 0;
  while (pos < text.size()) {
    bool matched = false;

    // Check for special token match at current position
    for (const auto &[token, id] : special_tokens) {
      if (pos + token.size() <= text.size() &&
          text.compare(pos, token.size(), token) == 0) {
        result.push_back(id);
        pos += token.size();
        matched = true;
        break;
      }
    }

    if (matched) continue;

    // No special token match - process as regular text
    // Find next special token position to limit the word
    size_t word_end = text.size();
    for (const auto &[token, id] : special_tokens) {
      size_t found = text.find(token, pos);
      if (found != std::string::npos && found < word_end) {
        word_end = found;
      }
    }

    // Process the text from pos to word_end
    if (word_end > pos) {
      std::string word = text.substr(pos, word_end - pos);
      std::string bl;
      bl.reserve(word.size() * 2);
      for (unsigned char c : word) {
        bl += byte_to_unicode_[c];
      }

      auto bpe_toks = BpeEncodeWithCache(bl, merges_rank_, &bpe_cache_);
      for (const std::string &tok : bpe_toks) {
        auto it = token2id_.find(tok);
        if (it != token2id_.end()) {
          result.push_back(it->second);
        }
      }
    }

    pos = word_end;
  }

  return result;
}

std::string QwenAsrTokenizer::Decode(const std::vector<int64_t> &token_ids) {
  std::string result;
  std::string buf;

  for (int64_t id : token_ids) {
    if (id >= 0 && static_cast<size_t>(id) < id2token_.size()) {
      const std::string &t = id2token_[id];
      if (IsSpecialTokenString(t)) {
        if (!buf.empty()) {
          result += DecodeBytesToUtf8(buf);
          buf.clear();
        }
        result += t;
      } else {
        buf += t;
      }
    }
  }

  if (!buf.empty()) {
    result += DecodeBytesToUtf8(buf);
    buf.clear();
  }

  return result;
}

std::string QwenAsrTokenizer::GetTokenStringStreaming(int64_t token_id,
                                                      std::string *state) const {
  if (token_id >= 0 && static_cast<size_t>(token_id) < id2token_.size()) {
    const std::string &t = id2token_[token_id];
    if (IsSpecialTokenString(t)) {
      return t;
    }

    std::string bytes = DecodeBytesToUtf8(t);
    if (!state) {
      return bytes;
    }

    state->append(bytes);

    std::string out;
    size_t i = 0;
    size_t last_good = 0;
    while (i < state->size()) {
      uint32_t cp = 0;
      size_t nb = 0;
      size_t pos0 = i;
      if (Utf8Next(*state, &i, &cp, &nb)) {
        last_good = i;
      } else {
        break;
      }
      (void)pos0;
    }

    if (last_good > 0) {
      out.assign(state->data(), last_good);
      state->erase(0, last_good);
    }

    return out;
  }
  return "";
}

}  // namespace sherpa_onnx
