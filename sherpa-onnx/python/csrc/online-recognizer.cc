// sherpa-onnx/python/csrc/online-recongizer.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/python/csrc/online-recognizer.h"

#include <string>
#include <vector>

#include "sherpa-onnx/csrc/online-recognizer.h"

namespace sherpa_onnx {

static void PybindOnlineRecognizerResult(py::module *m) {
  using PyClass = OnlineRecognizerResult;
  py::class_<PyClass>(*m, "OnlineRecognizerResult")
      .def_property_readonly("text", [](PyClass &self) { return self.text; });
}

static void PybindOnlineRecognizerConfig(py::module *m) {
  using PyClass = OnlineRecognizerConfig;
  py::class_<PyClass>(*m, "OnlineRecognizerConfig")
      .def(py::init<const FeatureExtractorConfig &,
                    const OnlineTransducerModelConfig &, const std::string &>(),
           py::arg("feat_config"), py::arg("model_config"), py::arg("tokens"))
      .def_readwrite("feat_config", &PyClass::feat_config)
      .def_readwrite("model_config", &PyClass::model_config)
      .def_readwrite("tokens", &PyClass::tokens)
      .def("__str__", &PyClass::ToString);
}

void PybindOnlineRecognizer(py::module *m) {
  PybindOnlineRecognizerResult(m);
  PybindOnlineRecognizerConfig(m);

  using PyClass = OnlineRecognizer;
  py::class_<PyClass>(*m, "OnlineRecognizer")
      .def(py::init<const OnlineRecognizerConfig &>(), py::arg("config"))
      .def("create_stream", &PyClass::CreateStream)
      .def("is_ready", &PyClass::IsReady)
      .def("decode_stream", &PyClass::DecodeStream)
      .def("decode_streams",
           [](PyClass &self, std::vector<OnlineStream *> ss) {
             self.DecodeStreams(ss.data(), ss.size());
           })
      .def("get_result", &PyClass::GetResult);
}

}  // namespace sherpa_onnx
