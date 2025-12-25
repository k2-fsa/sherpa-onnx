// sherpa-onnx/csrc/funasr-nano-tokenizer.cc

#include "sherpa-onnx/csrc/funasr-nano-tokenizer.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {

namespace {

static std::string FindTokenizerJson(const std::string &tokenizer_dir) {
  std::string p = tokenizer_dir + "/tokenizer.json";
  if (FileExists(p)) return p;
  return "";
}

static std::string FindVocabJson(const std::string &tokenizer_dir) {
  std::string p = tokenizer_dir + "/vocab.json";
  if (FileExists(p)) return p;
  return "";
}

static std::string FindMergesTxt(const std::string &tokenizer_dir) {
  std::string p = tokenizer_dir + "/merges.txt";
  if (FileExists(p)) return p;
  return "";
}

static std::string LoadBytesFromFile(const std::string &path) {
  std::vector<char> data = ReadFile(path);
  if (data.empty()) return "";
  return std::string(data.data(), data.size());
}

#if __ANDROID_API__ >= 9
static std::string LoadBytesFromFile(AAssetManager *mgr,
                                     const std::string &path) {
  std::vector<char> data = ReadFile(mgr, path);
  if (data.empty()) return "";
  return std::string(data.data(), data.size());
}
#endif

#if __OHOS__
static std::string LoadBytesFromFile(NativeResourceManager *mgr,
                                     const std::string &path) {
  std::vector<char> data = ReadFile(mgr, path);
  if (data.empty()) return "";
  return std::string(data.data(), data.size());
}
#endif

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

static inline void AppendUtf8(uint32_t cp, std::string *out) {
  if (!out) return;
  if (cp <= 0x7Fu) {
    out->push_back(static_cast<char>(cp));
  } else if (cp <= 0x7FFu) {
    out->push_back(static_cast<char>(0xC0u | ((cp >> 6) & 0x1Fu)));
    out->push_back(static_cast<char>(0x80u | (cp & 0x3Fu)));
  } else if (cp <= 0xFFFFu) {
    out->push_back(static_cast<char>(0xE0u | ((cp >> 12) & 0x0Fu)));
    out->push_back(static_cast<char>(0x80u | ((cp >> 6) & 0x3Fu)));
    out->push_back(static_cast<char>(0x80u | (cp & 0x3Fu)));
  } else {
    out->push_back(static_cast<char>(0xF0u | ((cp >> 18) & 0x07u)));
    out->push_back(static_cast<char>(0x80u | ((cp >> 12) & 0x3Fu)));
    out->push_back(static_cast<char>(0x80u | ((cp >> 6) & 0x3Fu)));
    out->push_back(static_cast<char>(0x80u | (cp & 0x3Fu)));
  }
}

static inline bool Utf8Next(const std::string &s, size_t *i, uint32_t *cp,
                            size_t *nbytes) {
  if (!i || !cp || !nbytes) return false;
  if (*i >= s.size()) return false;
  const unsigned char c = static_cast<unsigned char>(s[*i]);
  if (c < 0x80) {
    *cp = c;
    *nbytes = 1;
    return true;
  }
  if ((c >> 5) == 0x6) {  // 110xxxxx
    if (*i + 1 >= s.size()) return false;
    const unsigned char c1 = static_cast<unsigned char>(s[*i + 1]);
    if ((c1 >> 6) != 0x2) return false;
    *cp = ((c & 0x1F) << 6) | (c1 & 0x3F);
    *nbytes = 2;
    return true;
  }
  if ((c >> 4) == 0xE) {  // 1110xxxx
    if (*i + 2 >= s.size()) return false;
    const unsigned char c1 = static_cast<unsigned char>(s[*i + 1]);
    const unsigned char c2 = static_cast<unsigned char>(s[*i + 2]);
    if ((c1 >> 6) != 0x2 || (c2 >> 6) != 0x2) return false;
    *cp = ((c & 0x0F) << 12) | ((c1 & 0x3F) << 6) | (c2 & 0x3F);
    *nbytes = 3;
    return true;
  }
  if ((c >> 3) == 0x1E) {  // 11110xxx
    if (*i + 3 >= s.size()) return false;
    const unsigned char c1 = static_cast<unsigned char>(s[*i + 1]);
    const unsigned char c2 = static_cast<unsigned char>(s[*i + 2]);
    const unsigned char c3 = static_cast<unsigned char>(s[*i + 3]);
    if ((c1 >> 6) != 0x2 || (c2 >> 6) != 0x2 || (c3 >> 6) != 0x2) return false;
    *cp = ((c & 0x07) << 18) | ((c1 & 0x3F) << 12) | ((c2 & 0x3F) << 6) |
          (c3 & 0x3F);
    *nbytes = 4;
    return true;
  }
  return false;
}

static inline bool IsNewline(uint32_t cp) { return cp == '\n' || cp == '\r'; }

static inline bool IsAsciiSpace(uint32_t cp) { return cp == ' '; }

static inline bool IsWhitespace(uint32_t cp) {
  return cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r' || cp == '\v' ||
         cp == '\f';
}

static inline bool IsAsciiAlpha(uint32_t cp) {
  return (cp >= 'a' && cp <= 'z') || (cp >= 'A' && cp <= 'Z');
}

static inline bool IsAsciiDigit(uint32_t cp) { return (cp >= '0' && cp <= '9'); }

// A light-weight unicode letter/number approximation good enough for
// Qwen3(English/Chinese/Japanese/Korean + common scripts).
static inline bool IsLetter(uint32_t cp) {
  if (IsAsciiAlpha(cp)) return true;

  // CJK Unified Ideographs
  if (cp >= 0x4E00 && cp <= 0x9FFF) return true;
  // CJK Extension A
  if (cp >= 0x3400 && cp <= 0x4DBF) return true;
  // Hiragana/Katakana
  if (cp >= 0x3040 && cp <= 0x30FF) return true;
  // Hangul syllables
  if (cp >= 0xAC00 && cp <= 0xD7AF) return true;
  // Hangul Jamo
  if (cp >= 0x1100 && cp <= 0x11FF) return true;

  // Latin-1 Supplement + Latin Extended (covers most European letters)
  if (cp >= 0x00C0 && cp <= 0x02AF) return true;

  return false;
}

static inline bool IsNumber(uint32_t cp) {
  if (IsAsciiDigit(cp)) return true;
  // Fullwidth digits
  if (cp >= 0xFF10 && cp <= 0xFF19) return true;
  return false;
}

class JsonReader {
 public:
  explicit JsonReader(const std::string &s) : s_(s), p_(0) {}

  bool SeekToKey(const std::string &key) {
    std::string needle = "\"" + key + "\"";
    size_t pos = s_.find(needle);
    if (pos == std::string::npos) return false;
    p_ = pos + needle.size();
    return true;
  }

  void SkipWs() {
    while (p_ < s_.size()) {
      char c = s_[p_];
      if (c == ' ' || c == '\t' || c == '\r' || c == '\n') {
        ++p_;
      } else {
        break;
      }
    }
  }

  bool Consume(char c) {
    SkipWs();
    if (p_ < s_.size() && s_[p_] == c) {
      ++p_;
      return true;
    }
    return false;
  }

  bool Peek(char *c) const {
    if (!c) return false;
    size_t q = p_;
    while (q < s_.size()) {
      char x = s_[q];
      if (x == ' ' || x == '\t' || x == '\r' || x == '\n') {
        ++q;
        continue;
      }
      *c = x;
      return true;
    }
    return false;
  }

  bool ParseString(std::string *out) {
    if (!out) return false;
    SkipWs();
    if (p_ >= s_.size() || s_[p_] != '"') return false;
    ++p_;
    std::string r;
    while (p_ < s_.size()) {
      char c = s_[p_++];
      if (c == '"') {
        *out = std::move(r);
        return true;
      }
      if (c != '\\') {
        r.push_back(c);
        continue;
      }
      if (p_ >= s_.size()) return false;
      char esc = s_[p_++];
      switch (esc) {
        case '"':
          r.push_back('"');
          break;
        case '\\':
          r.push_back('\\');
          break;
        case '/':
          r.push_back('/');
          break;
        case 'b':
          r.push_back('\b');
          break;
        case 'f':
          r.push_back('\f');
          break;
        case 'n':
          r.push_back('\n');
          break;
        case 'r':
          r.push_back('\r');
          break;
        case 't':
          r.push_back('\t');
          break;
        case 'u': {
          if (p_ + 4 > s_.size()) return false;
          uint32_t u = 0;
          for (int i = 0; i < 4; ++i) {
            char h = s_[p_++];
            u <<= 4;
            if (h >= '0' && h <= '9')
              u |= (h - '0');
            else if (h >= 'a' && h <= 'f')
              u |= (h - 'a' + 10);
            else if (h >= 'A' && h <= 'F')
              u |= (h - 'A' + 10);
            else
              return false;
          }
          if (u >= 0xD800 && u <= 0xDBFF) {
            size_t save = p_;
            if (p_ + 6 <= s_.size() && s_[p_] == '\\' && s_[p_ + 1] == 'u') {
              p_ += 2;
              uint32_t v = 0;
              for (int i = 0; i < 4; ++i) {
                char h = s_[p_++];
                v <<= 4;
                if (h >= '0' && h <= '9')
                  v |= (h - '0');
                else if (h >= 'a' && h <= 'f')
                  v |= (h - 'a' + 10);
                else if (h >= 'A' && h <= 'F')
                  v |= (h - 'A' + 10);
                else
                  return false;
              }
              if (v >= 0xDC00 && v <= 0xDFFF) {
                uint32_t cp = 0x10000 + (((u - 0xD800) << 10) | (v - 0xDC00));
                AppendUtf8(cp, &r);
                break;
              }
            }
            p_ = save;
          }
          AppendUtf8(u, &r);
          break;
        }
        default:
          return false;
      }
    }
    return false;
  }

  bool ParseBool(bool *out) {
    if (!out) return false;
    SkipWs();
    if (p_ + 4 <= s_.size() && s_.compare(p_, 4, "true") == 0) {
      p_ += 4;
      *out = true;
      return true;
    }
    if (p_ + 5 <= s_.size() && s_.compare(p_, 5, "false") == 0) {
      p_ += 5;
      *out = false;
      return true;
    }
    return false;
  }

  bool ParseInt64(int64_t *out) {
    if (!out) return false;
    SkipWs();
    if (p_ >= s_.size()) return false;
    bool neg = false;
    if (s_[p_] == '-') {
      neg = true;
      ++p_;
    }
    if (p_ >= s_.size() || !std::isdigit(static_cast<unsigned char>(s_[p_]))) {
      return false;
    }
    int64_t v = 0;
    while (p_ < s_.size() &&
           std::isdigit(static_cast<unsigned char>(s_[p_]))) {
      int d = s_[p_] - '0';
      if (v > (std::numeric_limits<int64_t>::max() - d) / 10) return false;
      v = v * 10 + d;
      ++p_;
    }
    *out = neg ? -v : v;
    return true;
  }

  bool SkipValue() {
    SkipWs();
    if (p_ >= s_.size()) return false;
    char c = s_[p_];
    if (c == '"') {
      std::string tmp;
      return ParseString(&tmp);
    }
    if (c == '{') return SkipObject();
    if (c == '[') return SkipArray();
    if (c == 't' || c == 'f') {
      bool b = false;
      return ParseBool(&b);
    }
    if (c == 'n') {
      if (p_ + 4 <= s_.size() && s_.compare(p_, 4, "null") == 0) {
        p_ += 4;
        return true;
      }
      return false;
    }
    int64_t v = 0;
    return ParseInt64(&v);
  }

 private:
  bool SkipObject() {
    if (!Consume('{')) return false;
    SkipWs();
    if (Consume('}')) return true;
    while (true) {
      std::string k;
      if (!ParseString(&k)) return false;
      if (!Consume(':')) return false;
      if (!SkipValue()) return false;
      SkipWs();
      if (Consume('}')) return true;
      if (!Consume(',')) return false;
    }
  }

  bool SkipArray() {
    if (!Consume('[')) return false;
    SkipWs();
    if (Consume(']')) return true;
    while (true) {
      if (!SkipValue()) return false;
      SkipWs();
      if (Consume(']')) return true;
      if (!Consume(',')) return false;
    }
  }

 private:
  const std::string &s_;
  size_t p_;
};

namespace {
static inline int64_t TokenToIdOrDefault(
    const std::unordered_map<std::string, int32_t> &vocab,
    const std::string &tok, int64_t def_val) {
  auto it = vocab.find(tok);
  if (it == vocab.end()) return def_val;
  return static_cast<int64_t>(it->second);
}
}  // namespace

// Build bytes_to_unicode mapping (ByteLevel encoder/decoder).
static void BuildBytesToUnicode(
    std::string byte_to_unicode[256],
    std::unordered_map<std::string, uint8_t> *unicode_to_byte) {
  std::vector<uint32_t> bs;
  bs.reserve(256);
  for (uint32_t c = 33; c <= 126; ++c) bs.push_back(c);
  for (uint32_t c = 161; c <= 172; ++c) bs.push_back(c);
  for (uint32_t c = 174; c <= 255; ++c) bs.push_back(c);

  std::vector<uint32_t> cs = bs;
  cs.reserve(256);
  uint32_t n = 0;
  auto contains = [&](uint32_t b) -> bool {
    return std::find(bs.begin(), bs.end(), b) != bs.end();
  };
  for (uint32_t b = 0; b <= 255; ++b) {
    if (!contains(b)) {
      bs.push_back(b);
      cs.push_back(256 + n);
      ++n;
    }
  }

  if (unicode_to_byte) unicode_to_byte->clear();
  for (size_t i = 0; i < bs.size(); ++i) {
    uint32_t b = bs[i];
    uint32_t c = cs[i];
    std::string u;
    AppendUtf8(c, &u);
    byte_to_unicode[b] = u;
    if (unicode_to_byte) {
      (*unicode_to_byte)[u] = static_cast<uint8_t>(b);
    }
  }
}

// Parse vocab.json: {"token": id, ...}
static bool ParseVocabJson(const std::string &blob,
                           std::unordered_map<std::string, int32_t> *out) {
  if (!out) return false;
  out->clear();
  JsonReader r(blob);
  r.SkipWs();
  if (!r.Consume('{')) return false;
  r.SkipWs();
  if (r.Consume('}')) return true;

  while (true) {
    std::string key;
    if (!r.ParseString(&key)) return false;
    if (!r.Consume(':')) return false;
    int64_t id64 = 0;
    if (!r.ParseInt64(&id64)) return false;
    if (id64 < 0 || id64 > std::numeric_limits<int32_t>::max()) return false;
    (*out)[key] = static_cast<int32_t>(id64);

    r.SkipWs();
    if (r.Consume('}')) return true;
    if (!r.Consume(',')) return false;
  }
}

// Parse merges.txt: each non-comment line: "left right"
static bool ParseMergesTxt(const std::string &blob,
                           std::unordered_map<std::string, int32_t> *out) {
  if (!out) return false;
  out->clear();
  std::istringstream is(blob);
  std::string line;
  int32_t rank = 0;
  while (std::getline(is, line)) {
    if (line.empty()) continue;
    if (line.rfind("#version", 0) == 0) continue;
    std::string left, right;
    {
      std::istringstream ls(line);
      if (!(ls >> left >> right)) continue;
    }
    std::string key = left;
    key.push_back('\t');
    key.append(right);
    (*out)[key] = rank++;
  }
  return true;
}

static inline bool IsWordChar(uint32_t cp) {
  return IsLetter(cp) || IsNumber(cp) || cp == '_';
}

// A manual approximation for Qwen3 tokenizer Split regex.
// The regex is in tokenizer.json pre_tokenizer Split. We avoid std::regex
// due to missing \p{L}/\p{N} support in libc++/libstdc++ regex.
static std::vector<std::string> SplitByQwen3Pattern(const std::string &text) {
  std::vector<std::string> out;
  out.reserve(text.size() / 2 + 1);

  size_t i = 0;
  while (i < text.size()) {
    if (text[i] == '\'') {
      auto lower = [](char c) -> char {
        return static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
      };
      if (i + 1 < text.size()) {
        char c1 = lower(text[i + 1]);
        if (c1 == 's' || c1 == 't' || c1 == 'm' || c1 == 'd') {
          out.push_back(text.substr(i, 2));
          i += 2;
          continue;
        }
        if (i + 2 < text.size()) {
          char c2 = lower(text[i + 2]);
          if (c1 == 'r' && c2 == 'e') {
            out.push_back(text.substr(i, 3));
            i += 3;
            continue;
          }
          if (c1 == 'v' && c2 == 'e') {
            out.push_back(text.substr(i, 3));
            i += 3;
            continue;
          }
          if (c1 == 'l' && c2 == 'l') {
            out.push_back(text.substr(i, 3));
            i += 3;
            continue;
          }
        }
      }
    }

    size_t cur = i;
    uint32_t cp = 0;
    size_t n = 0;
    if (!Utf8Next(text, &cur, &cp, &n) || n == 0) {
      out.push_back(text.substr(i, 1));
      i += 1;
      continue;
    }

    auto peek_next_cp = [&](size_t pos, uint32_t *cp2, size_t *n2) -> bool {
      size_t t = pos;
      uint32_t x = 0;
      size_t nn = 0;
      if (!Utf8Next(text, &t, &x, &nn)) return false;
      if (cp2) *cp2 = x;
      if (n2) *n2 = nn;
      return true;
    };

    {
      uint32_t next_cp = 0;
      size_t next_n = 0;
      bool has_next = peek_next_cp(i + n, &next_cp, &next_n);

      bool cur_ok_prefix = (!IsNewline(cp) && !IsLetter(cp) && !IsNumber(cp));
      bool cur_is_letter = IsLetter(cp);

      if (cur_is_letter || (cur_ok_prefix && has_next && IsLetter(next_cp))) {
        size_t start = i;
        size_t j = i;
        if (!cur_is_letter) {
          j += n;
          while (j < text.size()) {
            size_t t = j;
            uint32_t cpl = 0;
            size_t nl = 0;
            if (!Utf8Next(text, &t, &cpl, &nl)) break;
            if (!IsLetter(cpl)) break;
            j += nl;
          }
        } else {
          j = i;
          while (j < text.size()) {
            size_t t = j;
            uint32_t cpl = 0;
            size_t nl = 0;
            if (!Utf8Next(text, &t, &cpl, &nl)) break;
            if (!IsLetter(cpl)) break;
            j += nl;
          }
        }
        out.push_back(text.substr(start, j - start));
        i = j;
        continue;
      }
    }

    if (IsNumber(cp)) {
      out.push_back(text.substr(i, n));
      i += n;
      continue;
    }

    {
      bool starts_with_space_prefix = IsAsciiSpace(cp);
      size_t start = i;
      size_t j = i;

      auto is_punct_like = [&](uint32_t x) -> bool {
        return (!IsWhitespace(x) && !IsLetter(x) && !IsNumber(x));
      };

      if (starts_with_space_prefix) {
        uint32_t next_cp = 0;
        size_t next_n = 0;
        if (peek_next_cp(i + n, &next_cp, &next_n) && is_punct_like(next_cp)) {
          j += n;
          while (j < text.size()) {
            size_t t = j;
            uint32_t cx = 0;
            size_t nx = 0;
            if (!Utf8Next(text, &t, &cx, &nx)) break;
            if (!is_punct_like(cx)) break;
            j += nx;
          }
          while (j < text.size()) {
            size_t t = j;
            uint32_t cx = 0;
            size_t nx = 0;
            if (!Utf8Next(text, &t, &cx, &nx)) break;
            if (!IsNewline(cx)) break;
            j += nx;
          }
          out.push_back(text.substr(start, j - start));
          i = j;
          continue;
        }
      } else if (is_punct_like(cp)) {
        while (j < text.size()) {
          size_t t = j;
          uint32_t cx = 0;
          size_t nx = 0;
          if (!Utf8Next(text, &t, &cx, &nx)) break;
            if (!is_punct_like(cx)) break;
            j += nx;
          }
          while (j < text.size()) {
          size_t t = j;
          uint32_t cx = 0;
          size_t nx = 0;
          if (!Utf8Next(text, &t, &cx, &nx)) break;
          if (!IsNewline(cx)) break;
          j += nx;
        }
        out.push_back(text.substr(start, j - start));
        i = j;
        continue;
      }
    }

    {
      if (IsWhitespace(cp)) {
        size_t start = i;
        size_t j = i;

        bool saw_newline = false;
        while (j < text.size()) {
          size_t t = j;
          uint32_t cx = 0;
          size_t nx = 0;
          if (!Utf8Next(text, &t, &cx, &nx)) break;
          if (IsNewline(cx)) {
            saw_newline = true;
            break;
          }
          if (!IsWhitespace(cx)) break;
          j += nx;
        }

        if (saw_newline) {
          while (j < text.size()) {
            size_t t = j;
            uint32_t cx = 0;
            size_t nx = 0;
            if (!Utf8Next(text, &t, &cx, &nx)) break;
            if (!IsNewline(cx)) break;
            j += nx;
          }
          out.push_back(text.substr(start, j - start));
          i = j;
          continue;
        }
      }
    }

    if (IsWhitespace(cp)) {
      bool only_ws_to_end = true;
      size_t j = i;
      while (j < text.size()) {
        size_t t = j;
        uint32_t cx = 0;
        size_t nx = 0;
        if (!Utf8Next(text, &t, &cx, &nx)) break;
        if (!IsWhitespace(cx)) {
          only_ws_to_end = false;
          break;
        }
        j += nx;
      }
      if (only_ws_to_end) {
        out.push_back(text.substr(i));
        break;
      }
    }

    if (IsWhitespace(cp)) {
      size_t start = i;
      size_t j = i;
      while (j < text.size()) {
        size_t t = j;
        uint32_t cx = 0;
        size_t nx = 0;
        if (!Utf8Next(text, &t, &cx, &nx)) break;
        if (!IsWhitespace(cx)) break;
        j += nx;
      }
      out.push_back(text.substr(start, j - start));
      i = j;
      continue;
    }

    out.push_back(text.substr(i, n));
    i += n;
  }

  return out;
}

static std::vector<std::string> SplitUtf8ToChars(const std::string &s) {
  std::vector<std::string> out;
  out.reserve(s.size());
  size_t i = 0;
  while (i < s.size()) {
    size_t t = i;
    uint32_t cp = 0;
    size_t n = 0;
    if (!Utf8Next(s, &t, &cp, &n) || n == 0) {
      out.push_back(s.substr(i, 1));
      i += 1;
      continue;
    }
    out.push_back(s.substr(i, n));
    i += n;
  }
  return out;
}

static inline std::string MakeMergeKey(const std::string &a,
                                       const std::string &b) {
  std::string k = a;
  k.push_back('\t');
  k.append(b);
  return k;
}

}  // namespace

// Parse tokenizer.json added_tokens: extract objects with {id, content, ...}
bool ParseAddedTokensFromTokenizerJson(
    const std::string &blob, std::vector<FunASRNanoTokenizer::AddedToken> *out) {
  if (!out) return false;
  out->clear();

  JsonReader r(blob);
  if (!r.SeekToKey("added_tokens")) return true;
  if (!r.Consume(':')) return false;
  if (!r.Consume('[')) return false;

  r.SkipWs();
  if (r.Consume(']')) return true;

  while (true) {
    if (!r.Consume('{')) return false;
    FunASRNanoTokenizer::AddedToken t;

    r.SkipWs();
    if (!r.Consume('}')) {
      while (true) {
        std::string k;
        if (!r.ParseString(&k)) return false;
        if (!r.Consume(':')) return false;

        if (k == "id") {
          int64_t v = 0;
          if (!r.ParseInt64(&v)) return false;
          t.id = static_cast<int32_t>(v);
        } else if (k == "content") {
          if (!r.ParseString(&t.content)) return false;
        } else if (k == "single_word") {
          if (!r.ParseBool(&t.single_word)) return false;
        } else if (k == "lstrip") {
          if (!r.ParseBool(&t.lstrip)) return false;
        } else if (k == "rstrip") {
          if (!r.ParseBool(&t.rstrip)) return false;
        } else if (k == "normalized") {
          if (!r.ParseBool(&t.normalized)) return false;
        } else if (k == "special") {
          if (!r.ParseBool(&t.special)) return false;
        } else {
          if (!r.SkipValue()) return false;
        }

        r.SkipWs();
        if (r.Consume('}')) break;
        if (!r.Consume(',')) return false;
      }
    }

    if (t.id >= 0 && !t.content.empty()) {
      out->push_back(std::move(t));
    }

    r.SkipWs();
    if (r.Consume(']')) return true;
    if (!r.Consume(',')) return false;
  }
}

// Build trie for AddedTokens longest match (byte-wise).
void BuildAddedTokensTrie(
    const std::vector<FunASRNanoTokenizer::AddedToken> &tokens,
    std::vector<FunASRNanoTokenizer::TrieNode> *trie) {
  if (!trie) return;
  trie->clear();
  trie->push_back(FunASRNanoTokenizer::TrieNode{});
  for (int32_t i = 0; i < static_cast<int32_t>(tokens.size()); ++i) {
    const auto &tok = tokens[i];
    int32_t node = 0;
    for (uint8_t b : std::vector<uint8_t>(tok.content.begin(),
                                          tok.content.end())) {
      auto it = (*trie)[node].next.find(b);
      if (it == (*trie)[node].next.end()) {
        int32_t new_node = static_cast<int32_t>(trie->size());
        trie->push_back(FunASRNanoTokenizer::TrieNode{});
        (*trie)[node].next.emplace(b, new_node);
        node = new_node;
      } else {
        node = it->second;
      }
    }
    (*trie)[node].token_index = i;
  }
}

void MergeVocabAndAddedTokens(
    std::unordered_map<std::string, int32_t> *vocab,
    const std::vector<FunASRNanoTokenizer::AddedToken> &added,
    std::unordered_set<std::string> *added_contents) {
  if (!vocab) return;
  if (added_contents) added_contents->clear();

  int32_t overwritten = 0;
  for (const auto &t : added) {
    if (t.id < 0 || t.content.empty()) continue;
    if (added_contents) added_contents->insert(t.content);

    auto it = vocab->find(t.content);
    if (it != vocab->end() && it->second != t.id) {
      ++overwritten;
    }
    (*vocab)[t.content] = t.id;
  }

  if (overwritten > 0) {
    SHERPA_ONNX_LOGE(
        "AddedTokens overwrote %d vocab entries with different ids. "
        "This is expected for some tokenizers; keeping added-token ids.",
        overwritten);
  }
}

void BuildIdToToken(
    const std::unordered_map<std::string, int32_t> &vocab,
    const std::unordered_set<std::string> &added_contents,
    std::vector<std::string> *id2token) {
  if (!id2token) return;
  int32_t max_id = -1;
  for (const auto &kv : vocab) {
    max_id = std::max(max_id, kv.second);
  }
  if (max_id < 0) {
    id2token->clear();
    return;
  }
  id2token->assign(static_cast<size_t>(max_id) + 1, std::string{});

  int32_t dup = 0;
  for (const auto &kv : vocab) {
    const std::string &tok = kv.first;
    int32_t id = kv.second;
    if (id < 0) continue;
    std::string &slot = (*id2token)[static_cast<size_t>(id)];
    if (slot.empty()) {
      slot = tok;
      continue;
    }
    if (slot == tok) continue;

    bool slot_is_added = added_contents.count(slot) > 0;
    bool tok_is_added = added_contents.count(tok) > 0;
    if (!slot_is_added && tok_is_added) {
      slot = tok;
    }
    ++dup;
  }

  if (dup > 0) {
    SHERPA_ONNX_LOGE(
        "Detected %d duplicated id->token collisions while building id2token. "
        "Kept added_tokens' string when possible.",
        dup);
  }
}


// Try to match an AddedToken at byte-position `pos`.
// Returns (matched_len_bytes, token_index) or (0, -1) if no match.
std::pair<int32_t, int32_t> MatchAddedToken(
    const std::string &text, size_t pos,
    const std::vector<FunASRNanoTokenizer::TrieNode> &trie) {
  if (trie.empty()) return {0, -1};
  int32_t node = 0;
  int32_t best_idx = -1;
  int32_t best_len = 0;

  size_t i = pos;
  while (i < text.size()) {
    uint8_t b = static_cast<uint8_t>(text[i]);
    auto it = trie[node].next.find(b);
    if (it == trie[node].next.end()) break;
    node = it->second;
    ++i;
    if (trie[node].token_index >= 0) {
      best_idx = trie[node].token_index;
      best_len = static_cast<int32_t>(i - pos);
    }
  }
  return {best_len, best_idx};
}

FunASRNanoTokenizer::FunASRNanoTokenizer(const std::string &tokenizer_dir) {
  Init(tokenizer_dir);
}

#if __ANDROID_API__ >= 9
FunASRNanoTokenizer::FunASRNanoTokenizer(AAssetManager *mgr,
                                         const std::string &tokenizer_dir) {
  Init(mgr, tokenizer_dir);
}
#endif

#if __OHOS__
FunASRNanoTokenizer::FunASRNanoTokenizer(NativeResourceManager *mgr,
                                         const std::string &tokenizer_dir) {
  Init(mgr, tokenizer_dir);
}
#endif


void FunASRNanoTokenizer::Init(const std::string &tokenizer_dir) {
  std::string tok_json = FindTokenizerJson(tokenizer_dir);
  if (tok_json.empty()) {
    SHERPA_ONNX_LOGE("Cannot find tokenizer.json in: %s",
                     tokenizer_dir.c_str());
    SHERPA_ONNX_EXIT(-1);
  }
  std::string vocab_json = FindVocabJson(tokenizer_dir);
  if (vocab_json.empty()) {
    SHERPA_ONNX_LOGE("Cannot find vocab.json in: %s", tokenizer_dir.c_str());
    SHERPA_ONNX_EXIT(-1);
  }
  std::string merges_txt = FindMergesTxt(tokenizer_dir);
  if (merges_txt.empty()) {
    SHERPA_ONNX_LOGE("Cannot find merges.txt in: %s", tokenizer_dir.c_str());
    SHERPA_ONNX_EXIT(-1);
  }

  const std::string tok_blob = LoadBytesFromFile(tok_json);
  const std::string vocab_blob = LoadBytesFromFile(vocab_json);
  const std::string merges_blob = LoadBytesFromFile(merges_txt);

  if (tok_blob.empty() || vocab_blob.empty() || merges_blob.empty()) {
    SHERPA_ONNX_LOGE("Failed to read tokenizer files from: %s",
                     tokenizer_dir.c_str());
    SHERPA_ONNX_EXIT(-1);
  }

  // Build ByteLevel bytes_to_unicode mapping
  BuildBytesToUnicode(byte_to_unicode_, &unicode_to_byte_);

  if (!ParseVocabJson(vocab_blob, &token2id_)) {
    SHERPA_ONNX_LOGE("Failed to parse vocab.json: %s", vocab_json.c_str());
    SHERPA_ONNX_EXIT(-1);
  }
  if (!ParseMergesTxt(merges_blob, &merges_rank_)) {
    SHERPA_ONNX_LOGE("Failed to parse merges.txt: %s", merges_txt.c_str());
    SHERPA_ONNX_EXIT(-1);
  }

  if (!ParseAddedTokensFromTokenizerJson(tok_blob, &added_tokens_)) {
    SHERPA_ONNX_LOGE("Failed to parse added_tokens from tokenizer.json: %s",
                     tok_json.c_str());
    SHERPA_ONNX_EXIT(-1);
  }
  MergeVocabAndAddedTokens(&token2id_, added_tokens_, &added_token_contents_);

  BuildIdToToken(token2id_, added_token_contents_, &id2token_);

  BuildAddedTokensTrie(added_tokens_, &trie_);

  FinalizeSpecialIds();
}

#if __ANDROID_API__ >= 9
void FunASRNanoTokenizer::Init(AAssetManager *mgr,
                               const std::string &tokenizer_dir) {
  std::string tok_json = tokenizer_dir + "/tokenizer.json";
  std::string vocab_json = tokenizer_dir + "/vocab.json";
  std::string merges_txt = tokenizer_dir + "/merges.txt";

  const std::string tok_blob = LoadBytesFromFile(mgr, tok_json);
  const std::string vocab_blob = LoadBytesFromFile(mgr, vocab_json);
  const std::string merges_blob = LoadBytesFromFile(mgr, merges_txt);

  if (tok_blob.empty() || vocab_blob.empty() || merges_blob.empty()) {
    SHERPA_ONNX_LOGE("Failed to read tokenizer files from assets: %s",
                     tokenizer_dir.c_str());
    SHERPA_ONNX_EXIT(-1);
  }

  BuildBytesToUnicode(byte_to_unicode_, &unicode_to_byte_);

  if (!ParseVocabJson(vocab_blob, &token2id_)) {
    SHERPA_ONNX_LOGE("Failed to parse vocab.json from assets: %s",
                     vocab_json.c_str());
    SHERPA_ONNX_EXIT(-1);
  }
  if (!ParseMergesTxt(merges_blob, &merges_rank_)) {
    SHERPA_ONNX_LOGE("Failed to parse merges.txt from assets: %s",
                     merges_txt.c_str());
    SHERPA_ONNX_EXIT(-1);
  }

  if (!ParseAddedTokensFromTokenizerJson(tok_blob, &added_tokens_)) {
    SHERPA_ONNX_LOGE("Failed to parse added_tokens from assets tokenizer.json");
    SHERPA_ONNX_EXIT(-1);
  }
  MergeVocabAndAddedTokens(&token2id_, added_tokens_, &added_token_contents_);
  BuildIdToToken(token2id_, added_token_contents_, &id2token_);
  BuildAddedTokensTrie(added_tokens_, &trie_);
  FinalizeSpecialIds();
}
#endif

#if __OHOS__
void FunASRNanoTokenizer::Init(NativeResourceManager *mgr,
                               const std::string &tokenizer_dir) {
  std::string tok_json = tokenizer_dir + "/tokenizer.json";
  std::string vocab_json = tokenizer_dir + "/vocab.json";
  std::string merges_txt = tokenizer_dir + "/merges.txt";

  const std::string tok_blob = LoadBytesFromFile(mgr, tok_json);
  const std::string vocab_blob = LoadBytesFromFile(mgr, vocab_json);
  const std::string merges_blob = LoadBytesFromFile(mgr, merges_txt);

  if (tok_blob.empty() || vocab_blob.empty() || merges_blob.empty()) {
    SHERPA_ONNX_LOGE("Failed to read tokenizer files from rawfile: %s",
                     tokenizer_dir.c_str());
    SHERPA_ONNX_EXIT(-1);
  }

  BuildBytesToUnicode(byte_to_unicode_, &unicode_to_byte_);

  if (!ParseVocabJson(vocab_blob, &token2id_)) {
    SHERPA_ONNX_LOGE("Failed to parse vocab.json from rawfile: %s",
                     vocab_json.c_str());
    SHERPA_ONNX_EXIT(-1);
  }
  if (!ParseMergesTxt(merges_blob, &merges_rank_)) {
    SHERPA_ONNX_LOGE("Failed to parse merges.txt from rawfile: %s",
                     merges_txt.c_str());
    SHERPA_ONNX_EXIT(-1);
  }

  if (!ParseAddedTokensFromTokenizerJson(tok_blob, &added_tokens_)) {
    SHERPA_ONNX_LOGE("Failed to parse added_tokens from rawfile tokenizer.json");
    SHERPA_ONNX_EXIT(-1);
  }
  MergeVocabAndAddedTokens(&token2id_, added_tokens_, &added_token_contents_);
  BuildIdToToken(token2id_, added_token_contents_, &id2token_);
  BuildAddedTokensTrie(added_tokens_, &trie_);
  FinalizeSpecialIds();
}
#endif

void FunASRNanoTokenizer::FinalizeSpecialIds() {
  im_end_token_id_ = TokenToIdOrDefault(token2id_, "<|im_end|>", 151645);
  eos_token_id_ = TokenToIdOrDefault(token2id_, "<|endoftext|>", -1);
  if (eos_token_id_ < 0) eos_token_id_ = im_end_token_id_;

  pad_token_id_ = TokenToIdOrDefault(token2id_, "<|pad|>", -1);
  if (pad_token_id_ < 0) pad_token_id_ = eos_token_id_;

  special_ids_.clear();
  special_ids_.insert(static_cast<int32_t>(eos_token_id_));
  special_ids_.insert(static_cast<int32_t>(im_end_token_id_));
  special_ids_.insert(static_cast<int32_t>(pad_token_id_));

  int64_t im_start = TokenToIdOrDefault(token2id_, "<|im_start|>", -1);
  if (im_start >= 0) special_ids_.insert(static_cast<int32_t>(im_start));
}

static inline bool CheckSingleWordBoundary(const std::string &text, size_t pos,
                                           size_t end) {
  auto prev_is_word = [&]() -> bool {
    if (pos == 0) return false;
    size_t j = pos;
    while (j > 0 && (static_cast<unsigned char>(text[j - 1]) & 0xC0) == 0x80)
      --j;
    if (j == 0) return false;
    size_t t = j - 1;
    while (t > 0 && (static_cast<unsigned char>(text[t]) & 0xC0) == 0x80) --t;
    size_t k = t;
    uint32_t cp = 0;
    size_t nb = 0;
    if (!Utf8Next(text, &k, &cp, &nb)) return false;
    return IsWordChar(cp);
  };

  auto next_is_word = [&]() -> bool {
    if (end >= text.size()) return false;
    size_t k = end;
    uint32_t cp = 0;
    size_t nb = 0;
    if (!Utf8Next(text, &k, &cp, &nb)) return false;
    return IsWordChar(cp);
  };

  return !(prev_is_word() || next_is_word());
}

// ByteLevel encode: map each byte to unicode char (bytes_to_unicode).
static inline std::string ByteLevelEncode(
    const std::string &token,
    const std::string byte_to_unicode[256]) {
  std::string out;
  out.reserve(token.size() * 2);
  for (unsigned char b : token) {
    out.append(byte_to_unicode[b]);
  }
  return out;
}

// BPE encode (with cache): bytelevel_word to merged token strings.
static std::vector<std::string> BpeEncodeWithCache(
    const std::string &word,
    const std::unordered_map<std::string, int32_t> &merges_rank,
    std::unordered_map<std::string, std::vector<std::string>> *cache) {
  if (!cache) return {};
  auto it = cache->find(word);
  if (it != cache->end()) return it->second;

  std::vector<std::string> symbols = SplitUtf8ToChars(word);
  if (symbols.empty()) {
    (*cache)[word] = {};
    return {};
  }
  if (symbols.size() == 1) {
    (*cache)[word] = symbols;
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

    // Merge best pair
    symbols[best_pos].append(symbols[best_pos + 1]);
    symbols.erase(symbols.begin() + best_pos + 1);
  }

  (*cache)[word] = symbols;
  return symbols;
}

std::vector<int64_t> FunASRNanoTokenizer::Encode(const std::string &text) {
  if (token2id_.empty()) {
    SHERPA_ONNX_LOGE("Tokenizer not initialized");
    SHERPA_ONNX_EXIT(-1);
  }

  std::vector<int64_t> out;
  if (text.empty()) return out;

  size_t pos = 0;
  size_t last = 0;
  while (pos < text.size()) {
    auto m = MatchAddedToken(text, pos, trie_);
    int32_t mlen = m.first;
    int32_t tidx = m.second;

    if (mlen > 0 && tidx >= 0) {
      const auto &tok = added_tokens_[static_cast<size_t>(tidx)];

      if (tok.single_word) {
        if (!CheckSingleWordBoundary(text, pos, pos + mlen)) {
          mlen = 0;
          tidx = -1;
        }
      }
    }

    if (mlen > 0 && tidx >= 0) {
      if (pos > last) {
        std::string seg = text.substr(last, pos - last);
        auto pieces = SplitByQwen3Pattern(seg);
        for (const auto &p : pieces) {
          std::string bl = ByteLevelEncode(p, byte_to_unicode_);
          auto bpe_toks =
              BpeEncodeWithCache(bl, merges_rank_, &bpe_cache_);
          for (const auto &bt : bpe_toks) {
            auto it = token2id_.find(bt);
            if (it == token2id_.end()) {
              continue;
            }
            out.push_back(static_cast<int64_t>(it->second));
          }
        }
      }

      const auto &atok = added_tokens_[static_cast<size_t>(tidx)];
      out.push_back(static_cast<int64_t>(atok.id));

      pos += static_cast<size_t>(mlen);
      last = pos;
      continue;
    }

    ++pos;
  }

  if (last < text.size()) {
    std::string seg = text.substr(last);
    auto pieces = SplitByQwen3Pattern(seg);
    for (const auto &p : pieces) {
      std::string bl = ByteLevelEncode(p, byte_to_unicode_);
      auto bpe_toks = BpeEncodeWithCache(bl, merges_rank_, &bpe_cache_);
      for (const auto &bt : bpe_toks) {
        auto it = token2id_.find(bt);
        if (it == token2id_.end()) continue;
        out.push_back(static_cast<int64_t>(it->second));
      }
    }
  }

  return out;
}

std::string FunASRNanoTokenizer::Decode(const std::vector<int64_t> &token_ids) {
  if (id2token_.empty()) {
    SHERPA_ONNX_LOGE("Tokenizer not initialized");
    SHERPA_ONNX_EXIT(-1);
  }
  if (token_ids.empty()) return "";

  std::vector<std::string> toks;
  toks.reserve(token_ids.size());
  for (int64_t v : token_ids) {
    if (v < 0) continue;
    if (v > static_cast<int64_t>(std::numeric_limits<int32_t>::max())) continue;
    int32_t id = static_cast<int32_t>(v);
    if (!special_ids_.empty() && special_ids_.count(id)) continue;
    if (id < 0 || static_cast<size_t>(id) >= id2token_.size()) continue;
    const std::string &t = id2token_[static_cast<size_t>(id)];
    if (!t.empty()) toks.push_back(t);
  }

  std::string merged;
  {
    size_t total = 0;
    for (const auto &t : toks) total += t.size();
    merged.reserve(total);
    for (const auto &t : toks) merged.append(t);
  }

  std::vector<uint8_t> bytes;
  bytes.reserve(merged.size());

  size_t i = 0;
  while (i < merged.size()) {
    size_t t = i;
    uint32_t cp = 0;
    size_t n = 0;
    if (!Utf8Next(merged, &t, &cp, &n) || n == 0) {
      bytes.push_back(static_cast<uint8_t>(merged[i]));
      i += 1;
      continue;
    }
    std::string ch = merged.substr(i, n);
    auto it = unicode_to_byte_.find(ch);
    if (it != unicode_to_byte_.end()) {
      bytes.push_back(it->second);
    } else {
      for (unsigned char b : ch) bytes.push_back(b);
    }
    i += n;
  }

  std::string out(reinterpret_cast<const char *>(bytes.data()), bytes.size());

  for (const char *sp : {"<|im_end|>", "<|im_start|>", "<|endoftext|>"}) {
    std::string needle(sp);
    size_t pos = 0;
    while ((pos = out.find(needle, pos)) != std::string::npos) {
      out.erase(pos, needle.size());
    }
  }

  TrimInPlace(&out);
  return out;
}

}  // namespace sherpa_onnx
