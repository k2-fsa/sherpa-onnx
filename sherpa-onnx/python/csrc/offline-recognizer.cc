// sherpa-onnx/python/csrc/offline-recognizer.cc
//
// Copyright (c)  2023 by manyeyes

#include "sherpa-onnx/python/csrc/offline-recognizer.h"

#include <string>
#include <vector>

#include "sherpa-onnx/csrc/offline-recognizer.h"

namespace sherpa_onnx {

static void PybindOfflineRecognizerConfig(py::module *m) {
  using PyClass = OfflineRecognizerConfig;
  py::class_<PyClass>(*m, "OfflineRecognizerConfig")
      .def(py::init<const OfflineFeatureExtractorConfig &,
                    const OfflineModelConfig &, const OfflineLMConfig &,
                    const std::string &, int32_t>(),
           py::arg("feat_config"), py::arg("model_config"),
           py::arg("lm_config") = OfflineLMConfig(),
           py::arg("decoding_method") = "greedy_search",
           py::arg("max_active_paths") = 4)
      .def_readwrite("feat_config", &PyClass::feat_config)
      .def_readwrite("model_config", &PyClass::model_config)
      .def_readwrite("lm_config", &PyClass::lm_config)
      .def_readwrite("decoding_method", &PyClass::decoding_method)
      .def_readwrite("max_active_paths", &PyClass::max_active_paths)
      .def("__str__", &PyClass::ToString);
}

void PybindOfflineRecognizer(py::module *m) {
  PybindOfflineRecognizerConfig(m);

  using PyClass = OfflineRecognizer;
  py::class_<PyClass>(*m, "OfflineRecognizer")
      .def(py::init<const OfflineRecognizerConfig &>(), py::arg("config"))
      .def("create_stream", &PyClass::CreateStream)
      .def("decode_stream", &PyClass::DecodeStream)
      .def("decode_streams",
           [](PyClass &self, std::vector<OfflineStream *> ss) {
             self.DecodeStreams(ss.data(), ss.size());
           });
}

}  // namespace sherpa_onnx
