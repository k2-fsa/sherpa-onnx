// sherpa-onnx/python/csrc/version.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa-onnx/python/csrc/version.h"

#include <string>

#include "sherpa-onnx/csrc/version.h"

namespace sherpa_onnx {

void PybindVersion(py::module *m) {
  m->attr("version") = std::string(GetVersionStr());

  m->attr("git_sha1") = std::string(GetGitSha1());

  m->attr("git_date") = std::string(GetGitDate());
}

}  // namespace sherpa_onnx
