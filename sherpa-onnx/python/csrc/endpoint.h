// sherpa-onnx/csrc/endpoint.h
//
// Copyright (c)  2023  Xiaomi Corporation

#ifndef SHERPA_ONNX_PYTHON_CSRC_ENDPOINT_H_
#define SHERPA_ONNX_PYTHON_CSRC_ENDPOINT_H_

#include "sherpa-onnx/python/csrc/sherpa-onnx.h"

namespace sherpa_onnx {

void PybindEndpoint(py::module *m);

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_PYTHON_CSRC_ENDPOINT_H_
