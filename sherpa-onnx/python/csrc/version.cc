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

  m->attr("__doc_version") = "str: The version of sherpa-onnx.";
  m->attr("__doc_git_sha1") =
      "str: The git commit SHA1 used to build sherpa-onnx.";
  m->attr("__doc_git_date") =
      "str: The git commit date used to build sherpa-onnx.";
}

}  // namespace sherpa_onnx
