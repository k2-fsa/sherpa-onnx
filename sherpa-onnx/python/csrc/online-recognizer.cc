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
                    const OnlineTransducerModelConfig &, const OnlineLMConfig &,
                    const EndpointConfig &, bool, const std::string &, int32_t,
                    float>(),
           py::arg("feat_config"), py::arg("model_config"),
           py::arg("lm_config") = OnlineLMConfig(), py::arg("endpoint_config"),
           py::arg("enable_endpoint"), py::arg("decoding_method"),
           py::arg("max_active_paths"), py::arg("context_score"))
      .def_readwrite("feat_config", &PyClass::feat_config)
      .def_readwrite("model_config", &PyClass::model_config)
      .def_readwrite("endpoint_config", &PyClass::endpoint_config)
      .def_readwrite("enable_endpoint", &PyClass::enable_endpoint)
      .def_readwrite("decoding_method", &PyClass::decoding_method)
      .def_readwrite("max_active_paths", &PyClass::max_active_paths)
      .def_readwrite("context_score", &PyClass::context_score)
      .def("__str__", &PyClass::ToString);
}

void PybindOnlineRecognizer(py::module *m) {
  PybindOnlineRecognizerResult(m);
  PybindOnlineRecognizerConfig(m);

  using PyClass = OnlineRecognizer;
  py::class_<PyClass>(*m, "OnlineRecognizer")
      .def(py::init<const OnlineRecognizerConfig &>(), py::arg("config"))
      .def("create_stream",
           [](const PyClass &self) { return self.CreateStream(); })
      .def(
          "create_stream",
          [](PyClass &self,
             const std::vector<std::vector<int32_t>> &contexts_list) {
            return self.CreateStream(contexts_list);
          },
          py::arg("contexts_list"))
      .def("is_ready", &PyClass::IsReady)
      .def("decode_stream", &PyClass::DecodeStream)
      .def("decode_streams",
           [](PyClass &self, std::vector<OnlineStream *> ss) {
             self.DecodeStreams(ss.data(), ss.size());
           })
      .def("get_result", &PyClass::GetResult)
      .def("is_endpoint", &PyClass::IsEndpoint)
      .def("reset", &PyClass::Reset);
}

}  // namespace sherpa_onnx
