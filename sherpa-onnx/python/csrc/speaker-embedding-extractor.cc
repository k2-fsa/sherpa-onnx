// sherpa-onnx/python/csrc/speaker-embedding-extractor.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/python/csrc/speaker-embedding-extractor.h"

#include "sherpa-onnx/csrc/speaker-embedding-extractor.h"

namespace sherpa_onnx {

static void PybindSpeakerEmbeddingExtractorConfig(py::module *m) {
  using PyClass = SpeakerEmbeddingExtractorConfig;
  py::class_<PyClass>(*m, "SpeakerEmbeddingExtractorConfig")
      .def(py::init<>())
      .def(py::init<const std::string &, int32_t, bool, const std::string>(),
           py::arg("model"), py::arg("num_threads") = 1,
           py::arg("debug") = false, py::arg("provider") = "cpu")
      .def_readwrite("model", &PyClass::model)
      .def_readwrite("num_threads", &PyClass::num_threads)
      .def_readwrite("debug", &PyClass::debug)
      .def_readwrite("provider", &PyClass::provider)
      .def("validate", &PyClass::Validate)
      .def("__str__", &PyClass::ToString);
}

void PybindSpeakerEmbeddingExtractor(py::module *m) {
  PybindSpeakerEmbeddingExtractorConfig(m);

  using PyClass = SpeakerEmbeddingExtractor;
  py::class_<PyClass>(*m, "SpeakerEmbeddingExtractor")
      .def(py::init<const SpeakerEmbeddingExtractorConfig &>(),
           py::arg("config"), py::call_guard<py::gil_scoped_release>())
      .def_property_readonly("dim", &PyClass::Dim)
      .def("create_stream", &PyClass::CreateStream)
      .def("compute", &PyClass::Compute);
}

}  // namespace sherpa_onnx
