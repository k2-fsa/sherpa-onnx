// sherpa-onnx/python/csrc/offline-tts.cc
//
// Copyright (c)  2023  Xiaomi Corporation
#include "sherpa-onnx/python/csrc/offline-tts.h"

#include <algorithm>
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
      .def(py::init<const OfflineTtsModelConfig &, const std::string &,
                    int32_t>(),
           py::arg("model"), py::arg("rule_fsts") = "",
           py::arg("max_num_sentences") = 2)
      .def_readwrite("model", &PyClass::model)
      .def_readwrite("rule_fsts", &PyClass::rule_fsts)
      .def_readwrite("max_num_sentences", &PyClass::max_num_sentences)
      .def("validate", &PyClass::Validate)
      .def("__str__", &PyClass::ToString);
}

void PybindOfflineTts(py::module *m) {
  PybindOfflineTtsConfig(m);
  PybindGeneratedAudio(m);

  using PyClass = OfflineTts;
  py::class_<PyClass>(*m, "OfflineTts")
      .def(py::init<const OfflineTtsConfig &>(), py::arg("config"),
           py::call_guard<py::gil_scoped_release>())
      .def_property_readonly("sample_rate", &PyClass::SampleRate)
      .def_property_readonly("num_speakers", &PyClass::NumSpeakers)
      .def(
          "generate",
          [](const PyClass &self, const std::string &text, int64_t sid,
             float speed, std::function<void(py::array_t<float>)> callback)
              -> GeneratedAudio {
            if (!callback) {
              return self.Generate(text, sid, speed);
            }

            std::function<void(const float *, int32_t)> callback_wrapper =
                [callback](const float *samples, int32_t n) {
                  // CAUTION(fangjun): we have to copy samples since it is
                  // freed once the call back returns.

                  pybind11::gil_scoped_acquire acquire;

                  pybind11::array_t<float> array(n);
                  py::buffer_info buf = array.request();
                  auto p = static_cast<float *>(buf.ptr);
                  std::copy(samples, samples + n, p);
                  callback(array);
                };

            return self.Generate(text, sid, speed, callback_wrapper);
          },
          py::arg("text"), py::arg("sid") = 0, py::arg("speed") = 1.0,
          py::arg("callback") = py::none(),
          py::call_guard<py::gil_scoped_release>());
}

}  // namespace sherpa_onnx
