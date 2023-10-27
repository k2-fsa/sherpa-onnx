// sherpa-onnx/csrc/text-utils.cc
//
// Copyright 2009-2011  Saarland University;  Microsoft Corporation
// Copyright      2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/text-utils.h"

#include <assert.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/macros.h"

// This file is copied/modified from
// https://github.com/kaldi-asr/kaldi/blob/master/src/util/text-utils.cc

namespace sherpa_onnx {

// copied from kaldi/src/util/text-util.cc
template <class T>
class NumberIstream {
 public:
  explicit NumberIstream(std::istream &i) : in_(i) {}

  NumberIstream &operator>>(T &x) {
    if (!in_.good()) return *this;
    in_ >> x;
    if (!in_.fail() && RemainderIsOnlySpaces()) return *this;
    return ParseOnFail(&x);
  }

 private:
  std::istream &in_;

  bool RemainderIsOnlySpaces() {
    if (in_.tellg() != std::istream::pos_type(-1)) {
      std::string rem;
      in_ >> rem;

      if (rem.find_first_not_of(' ') != std::string::npos) {
        // there is not only spaces
        return false;
      }
    }

    in_.clear();
    return true;
  }

  NumberIstream &ParseOnFail(T *x) {
    std::string str;
    in_.clear();
    in_.seekg(0);
    // If the stream is broken even before trying
    // to read from it or if there are many tokens,
    // it's pointless to try.
    if (!(in_ >> str) || !RemainderIsOnlySpaces()) {
      in_.setstate(std::ios_base::failbit);
      return *this;
    }

    std::unordered_map<std::string, T> inf_nan_map;
    // we'll keep just uppercase values.
    inf_nan_map["INF"] = std::numeric_limits<T>::infinity();
    inf_nan_map["+INF"] = std::numeric_limits<T>::infinity();
    inf_nan_map["-INF"] = -std::numeric_limits<T>::infinity();
    inf_nan_map["INFINITY"] = std::numeric_limits<T>::infinity();
    inf_nan_map["+INFINITY"] = std::numeric_limits<T>::infinity();
    inf_nan_map["-INFINITY"] = -std::numeric_limits<T>::infinity();
    inf_nan_map["NAN"] = std::numeric_limits<T>::quiet_NaN();
    inf_nan_map["+NAN"] = std::numeric_limits<T>::quiet_NaN();
    inf_nan_map["-NAN"] = -std::numeric_limits<T>::quiet_NaN();
    // MSVC
    inf_nan_map["1.#INF"] = std::numeric_limits<T>::infinity();
    inf_nan_map["-1.#INF"] = -std::numeric_limits<T>::infinity();
    inf_nan_map["1.#QNAN"] = std::numeric_limits<T>::quiet_NaN();
    inf_nan_map["-1.#QNAN"] = -std::numeric_limits<T>::quiet_NaN();

    std::transform(str.begin(), str.end(), str.begin(), ::toupper);

    if (inf_nan_map.find(str) != inf_nan_map.end()) {
      *x = inf_nan_map[str];
    } else {
      in_.setstate(std::ios_base::failbit);
    }

    return *this;
  }
};

/// ConvertStringToReal converts a string into either float or double
/// and returns false if there was any kind of problem (i.e. the string
/// was not a floating point number or contained extra non-whitespace junk).
/// Be careful- this function will successfully read inf's or nan's.
template <typename T>
bool ConvertStringToReal(const std::string &str, T *out) {
  std::istringstream iss(str);

  NumberIstream<T> i(iss);

  i >> *out;

  if (iss.fail()) {
    // Number conversion failed.
    return false;
  }

  return true;
}

template bool ConvertStringToReal<float>(const std::string &str, float *out);

template bool ConvertStringToReal<double>(const std::string &str, double *out);

void SplitStringToVector(const std::string &full, const char *delim,
                         bool omit_empty_strings,
                         std::vector<std::string> *out) {
  size_t start = 0, found = 0, end = full.size();
  out->clear();
  while (found != std::string::npos) {
    found = full.find_first_of(delim, start);
    // start != end condition is for when the delimiter is at the end
    if (!omit_empty_strings || (found != start && start != end))
      out->push_back(full.substr(start, found - start));
    start = found + 1;
  }
}

template <class F>
bool SplitStringToFloats(const std::string &full, const char *delim,
                         bool omit_empty_strings,  // typically false
                         std::vector<F> *out) {
  assert(out != nullptr);
  if (*(full.c_str()) == '\0') {
    out->clear();
    return true;
  }
  std::vector<std::string> split;
  SplitStringToVector(full, delim, omit_empty_strings, &split);
  out->resize(split.size());
  for (size_t i = 0; i < split.size(); ++i) {
    // assume atof never fails
    F f = 0;
    if (!ConvertStringToReal(split[i], &f)) return false;
    (*out)[i] = f;
  }
  return true;
}

// Instantiate the template above for float and double.
template bool SplitStringToFloats(const std::string &full, const char *delim,
                                  bool omit_empty_strings,
                                  std::vector<float> *out);
template bool SplitStringToFloats(const std::string &full, const char *delim,
                                  bool omit_empty_strings,
                                  std::vector<double> *out);

static bool IsPunct(char c) { return c != '\'' && std::ispunct(c); }
static bool IsGermanUmlauts(const std::string &words) {
  // ä 0xC3 0xA4
  // ö 0xC3 0xB6
  // ü 0xC3 0xBC
  // Ä 0xC3 0x84
  // Ö 0xC3 0x96
  // Ü 0xC3 0x9C
  // ß 0xC3 0x9F

  if (words.size() != 2 || static_cast<uint8_t>(words[0]) != 0xc3) {
    return false;
  }

  auto c = static_cast<uint8_t>(words[1]);
  if (c == 0xa4 || c == 0xb6 || c == 0xbC || c == 0x84 || c == 0x96 ||
      c == 0x9c || c == 0x9f) {
    return true;
  }

  return false;
}

static std::vector<std::string> MergeCharactersIntoWords(
    const std::vector<std::string> &words) {
  std::vector<std::string> ans;

  int32_t n = static_cast<int32_t>(words.size());
  int32_t i = 0;
  int32_t prev = -1;

  while (i < n) {
    const auto &w = words[i];
    if (w.size() >= 3 || (w.size() == 2 && !IsGermanUmlauts(w)) ||
        (w.size() == 1 && (IsPunct(w[0]) || std::isspace(w[0])))) {
      if (prev != -1) {
        std::string t;
        for (; prev < i; ++prev) {
          t.append(words[prev]);
        }
        prev = -1;
        ans.push_back(std::move(t));
      }

      if (!std::isspace(w[0])) {
        ans.push_back(w);
      }
      ++i;
      continue;
    }

    // e.g., öffnen
    if (w.size() == 1 || (w.size() == 2 && IsGermanUmlauts(w))) {
      if (prev == -1) {
        prev = i;
      }
      ++i;
      continue;
    }

    SHERPA_ONNX_LOGE("Ignore %s", w.c_str());
    ++i;
  }

  if (prev != -1) {
    std::string t;
    for (; prev < i; ++prev) {
      t.append(words[prev]);
    }
    ans.push_back(std::move(t));
  }

  return ans;
}

std::vector<std::string> SplitUtf8(const std::string &text) {
  const uint8_t *begin = reinterpret_cast<const uint8_t *>(text.c_str());
  const uint8_t *end = begin + text.size();

  // Note that English words are split into single characters.
  // We need to invoke MergeCharactersIntoWords() to merge them
  std::vector<std::string> ans;

  auto start = begin;
  while (start < end) {
    uint8_t c = *start;
    uint8_t i = 0x80;
    int32_t num_bytes = 0;

    // see
    // https://en.wikipedia.org/wiki/UTF-8
    for (; c & i; i >>= 1) {
      ++num_bytes;
    }

    if (num_bytes == 0) {
      // this is an ascii
      ans.emplace_back(reinterpret_cast<const char *>(start), 1);
      ++start;
    } else if (2 <= num_bytes && num_bytes <= 4) {
      ans.emplace_back(reinterpret_cast<const char *>(start), num_bytes);
      start += num_bytes;
    } else {
      SHERPA_ONNX_LOGE("Invalid byte at position: %d",
                       static_cast<int32_t>(start - begin));
      // skip this byte
      ++start;
    }
  }

  return MergeCharactersIntoWords(ans);
}

}  // namespace sherpa_onnx
