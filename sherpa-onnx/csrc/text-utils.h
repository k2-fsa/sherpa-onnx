// sherpa-onnx/csrc/text-utils.h
//
// Copyright 2009-2011  Saarland University;  Microsoft Corporation
// Copyright      2023  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_TEXT_UTILS_H_
#define SHERPA_ONNX_CSRC_TEXT_UTILS_H_
#include <stdlib.h>

#include <string>
#include <vector>

#ifdef _MSC_VER
#define SHERPA_ONNX_STRTOLL(cur_cstr, end_cstr) \
  _strtoi64(cur_cstr, end_cstr, 10);
#else
#define SHERPA_ONNX_STRTOLL(cur_cstr, end_cstr) strtoll(cur_cstr, end_cstr, 10);
#endif

// This file is copied/modified from
// https://github.com/kaldi-asr/kaldi/blob/master/src/util/text-utils.h

namespace sherpa_onnx {

/// Split a string using any of the single character delimiters.
/// If omit_empty_strings == true, the output will contain any
/// nonempty strings after splitting on any of the
/// characters in the delimiter.  If omit_empty_strings == false,
/// the output will contain n+1 strings if there are n characters
/// in the set "delim" within the input string.  In this case
/// the empty string is split to a single empty string.
void SplitStringToVector(const std::string &full, const char *delim,
                         bool omit_empty_strings,
                         std::vector<std::string> *out);

/**
  \brief Split a string (e.g. 1:2:3) into a vector of integers.

  \param [in]  delim  String containing a list of characters, any of which
                      is allowed as a delimiter.
  \param [in] omit_empty_strings If true, empty strings between delimiters are
                      allowed and will not produce an output integer; if false,
                      instances of characters in 'delim' that are consecutive or
                      at the start or end of the string would be an error.
                      You'll normally want this to be true if 'delim' consists
                      of spaces, and false otherwise.
  \param [out] out   The output list of integers.
*/
template <class I>
bool SplitStringToIntegers(const std::string &full, const char *delim,
                           bool omit_empty_strings,  // typically false [but
                                                     // should probably be true
                                                     // if "delim" is spaces].
                           std::vector<I> *out) {
  static_assert(std::is_integral<I>::value, "");
  if (*(full.c_str()) == '\0') {
    out->clear();
    return true;
  }
  std::vector<std::string> split;
  SplitStringToVector(full, delim, omit_empty_strings, &split);
  out->resize(split.size());
  for (size_t i = 0; i < split.size(); i++) {
    const char *this_str = split[i].c_str();
    char *end = NULL;
    int64_t j = 0;
    j = SHERPA_ONNX_STRTOLL(this_str, &end);
    if (end == this_str || *end != '\0') {
      out->clear();
      return false;
    } else {
      I jI = static_cast<I>(j);
      if (static_cast<int64_t>(jI) != j) {
        // output type cannot fit this integer.
        out->clear();
        return false;
      }
      (*out)[i] = jI;
    }
  }
  return true;
}

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_TEXT_UTILS_H_
