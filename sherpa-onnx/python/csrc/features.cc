// sherpa-onnx/python/csrc/features.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/python/csrc/features.h"

#include "sherpa-onnx/csrc/features.h"

namespace sherpa_onnx {

static void PybindFeatureExtractorConfig(py::module *m) {
  using PyClass = FeatureExtractorConfig;
  py::class_<PyClass>(*m, "FeatureExtractorConfig")
      .def(py::init<float, int32_t, int32_t>(),
           py::arg("sampling_rate") = 16000, py::arg("feature_dim") = 80,
           py::arg("max_feature_vectors") = -1)
      .def_readwrite("sampling_rate", &PyClass::sampling_rate)
      .def_readwrite("feature_dim", &PyClass::feature_dim)
      .def_readwrite("max_feature_vectors", &PyClass::max_feature_vectors)
      .def("__str__", &PyClass::ToString);
}

void PybindFeatures(py::module *m) { PybindFeatureExtractorConfig(m); }

}  // namespace sherpa_onnx
