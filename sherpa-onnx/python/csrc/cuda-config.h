// sherpa-onnx/python/csrc/cuda-config.h
//
// Copyright (c)  2024  Uniphore (Author: Manickavela A)

#ifndef SHERPA_ONNX_PYTHON_CSRC_CUDA_CONFIG_H_
#define SHERPA_ONNX_PYTHON_CSRC_CUDA_CONFIG_H_

#include "sherpa-onnx/python/csrc/sherpa-onnx.h"

namespace sherpa_onnx {

void PybindCudaConfig(py::module *m);

}

#endif  // SHERPA_ONNX_PYTHON_CSRC_CUDA_CONFIG_H_
