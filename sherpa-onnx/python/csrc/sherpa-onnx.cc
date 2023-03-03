// sherpa-onnx/python/csrc/sherpa-onnx.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/python/csrc/sherpa-onnx.h"

#include "sherpa-onnx/python/csrc/display.h"
#include "sherpa-onnx/python/csrc/endpoint.h"
#include "sherpa-onnx/python/csrc/features.h"
#include "sherpa-onnx/python/csrc/online-recognizer.h"
#include "sherpa-onnx/python/csrc/online-stream.h"
#include "sherpa-onnx/python/csrc/online-transducer-model-config.h"

namespace sherpa_onnx {

PYBIND11_MODULE(_sherpa_onnx, m) {
  m.doc() = "pybind11 binding of sherpa-onnx";
  PybindFeatures(&m);
  PybindOnlineTransducerModelConfig(&m);
  PybindOnlineStream(&m);
  PybindEndpoint(&m);
  PybindOnlineRecognizer(&m);

  PybindDisplay(&m);
}

}  // namespace sherpa_onnx
