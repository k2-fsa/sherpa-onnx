// sherpa-onnx/csrc/text-utils.cc
//
// Copyright 2009-2011  Saarland University;  Microsoft Corporation
// Copyright      2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/text-utils.h"

#include <assert.h>

#include <string>
#include <vector>

// This file is copied/modified from
// https://github.com/kaldi-asr/kaldi/blob/master/src/util/text-utils.cc

namespace sherpa_onnx {

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
    (*out)[i] = atof(split[i].c_str());
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

}  // namespace sherpa_onnx
