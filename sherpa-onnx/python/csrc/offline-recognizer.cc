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
                    const std::string &, int32_t, float>(),
           py::arg("feat_config"), py::arg("model_config"),
           py::arg("lm_config") = OfflineLMConfig(),
           py::arg("decoding_method") = "greedy_search",
           py::arg("max_active_paths") = 4, py::arg("context_score") = 1.5)
      .def_readwrite("feat_config", &PyClass::feat_config)
      .def_readwrite("model_config", &PyClass::model_config)
      .def_readwrite("lm_config", &PyClass::lm_config)
      .def_readwrite("decoding_method", &PyClass::decoding_method)
      .def_readwrite("max_active_paths", &PyClass::max_active_paths)
      .def_readwrite("context_score", &PyClass::context_score)
      .def("__str__", &PyClass::ToString);
}

void PybindOfflineRecognizer(py::module *m) {
  PybindOfflineRecognizerConfig(m);

  using PyClass = OfflineRecognizer;
  py::class_<PyClass>(*m, "OfflineRecognizer")
      .def(py::init<const OfflineRecognizerConfig &>(), py::arg("config"))
      .def("create_stream",
           [](const PyClass &self) { return self.CreateStream(); })
      .def(
          "create_stream",
          [](PyClass &self,
             const std::vector<std::vector<int32_t>> &contexts_list) {
            return self.CreateStream(contexts_list);
          },
          py::arg("contexts_list"))
      .def("decode_stream", &PyClass::DecodeStream)
      .def("decode_streams",
           [](const PyClass &self, std::vector<OfflineStream *> ss) {
             self.DecodeStreams(ss.data(), ss.size());
           });
}

}  // namespace sherpa_onnx
