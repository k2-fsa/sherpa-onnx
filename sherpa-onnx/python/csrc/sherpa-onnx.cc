// sherpa-onnx/python/csrc/sherpa-onnx.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/python/csrc/sherpa-onnx.h"

namespace sherpa_onnx {

PYBIND11_MODULE(_sherpa_ncnn, m) {
  m.doc() = "pybind11 binding of sherpa-onnx";
}

}  // namespace sherpa_onnx
