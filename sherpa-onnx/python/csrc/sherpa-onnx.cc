// sherpa-onnx/python/csrc/sherpa-onnx.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/python/csrc/sherpa-onnx.h"

#include "sherpa-onnx/python/csrc/features.h"

namespace sherpa_onnx {

PYBIND11_MODULE(_sherpa_onnx, m) {
  m.doc() = "pybind11 binding of sherpa-onnx";
  PybindFeatures(&m);
}

}  // namespace sherpa_onnx
