// sherpa-onnx/python/csrc/offline-tts.cc
//
// Copyright (c)  2023  Xiaomi Corporation
#include "sherpa-onnx/python/csrc/offline-tts.h"

#include <string>

#include "sherpa-onnx/csrc/offline-tts.h"
#include "sherpa-onnx/python/csrc/offline-tts-model-config.h"

namespace sherpa_onnx {

static void PybindGeneratedAudio(py::module *m) {
  using PyClass = GeneratedAudio;
  py::class_<PyClass>(*m, "GeneratedAudio")
      .def(py::init<>())
      .def_readwrite("samples", &PyClass::samples)
      .def_readwrite("sample_rate", &PyClass::sample_rate)
      .def("__str__", [](PyClass &self) {
        std::ostringstream os;
        os << "GeneratedAudio(sample_rate=" << self.sample_rate << ", ";
        os << "num_samples=" << self.samples.size() << ")";
        return os.str();
      });
}

static void PybindOfflineTtsConfig(py::module *m) {
  PybindOfflineTtsModelConfig(m);

  using PyClass = OfflineTtsConfig;
  py::class_<PyClass>(*m, "OfflineTtsConfig")
      .def(py::init<>())
      .def(py::init<const OfflineTtsModelConfig &, const std::string &>(),
           py::arg("model"), py::arg("rule_fsts") = "")
      .def_readwrite("model", &PyClass::model)
      .def_readwrite("rule_fsts", &PyClass::rule_fsts)
      .def("__str__", &PyClass::ToString);
}

void PybindOfflineTts(py::module *m) {
  PybindOfflineTtsConfig(m);
  PybindGeneratedAudio(m);

  using PyClass = OfflineTts;
  py::class_<PyClass>(*m, "OfflineTts")
      .def(py::init<const OfflineTtsConfig &>(), py::arg("config"))
      .def("generate", &PyClass::Generate, py::arg("text"), py::arg("sid") = 0,
           py::arg("speed") = 1.0, py::call_guard<py::gil_scoped_release>());
}

}  // namespace sherpa_onnx
