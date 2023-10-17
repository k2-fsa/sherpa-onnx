// sherpa-onnx/csrc/text-utils.cc
//
// Copyright 2009-2011  Saarland University;  Microsoft Corporation
// Copyright      2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/text-utils.h"

#include <assert.h>

#include <algorithm>
#include <cctype>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "source/utf8.h"

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

std::vector<std::string> SplitUtf8(const std::string &text) {
  char *begin = const_cast<char *>(text.c_str());
  char *end = begin + text.size();

  std::vector<std::string> ans;
  std::string buf;

  while (begin < end) {
    uint32_t code = utf8::next(begin, end);

    // 1. is punctuation
    if (std::ispunct(code)) {
      if (!buf.empty()) {
        ans.push_back(std::move(buf));
      }

      char s[5] = {0};
      utf8::append(code, s);
      ans.push_back(s);
      continue;
    }

    // 2. is space
    if (std::isspace(code)) {
      if (!buf.empty()) {
        ans.push_back(std::move(buf));
      }
      continue;
    }

    // 3. is alpha
    if (std::isalpha(code)) {
      buf.push_back(code);
      continue;
    }

    if (!buf.empty()) {
      ans.push_back(std::move(buf));
    }

    // for others

    char s[5] = {0};
    utf8::append(code, s);
    ans.push_back(s);
  }

  if (!buf.empty()) {
    ans.push_back(std::move(buf));
  }

  return ans;
}
}  // namespace sherpa_onnx
