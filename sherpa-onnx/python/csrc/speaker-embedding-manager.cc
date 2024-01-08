// sherpa-onnx/python/csrc/speaker-embedding-manager.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/python/csrc/speaker-embedding-manager.h"

#include <string>
#include <vector>

#include "sherpa-onnx/csrc/speaker-embedding-manager.h"
#include "sherpa-onnx/python/csrc/speaker-embedding-manager.h"

namespace sherpa_onnx {

void PybindSpeakerEmbeddingManager(py::module *m) {
  using PyClass = SpeakerEmbeddingManager;
  py::class_<PyClass>(*m, "SpeakerEmbeddingManager")
      .def(py::init<int32_t>(), py::arg("dim"))
      .def(
          "add",
          [](const PyClass &self, const std::string &name,
             const std::vector<float> &v) -> bool {
            return self.Add(name, v.data());
          },
          py::arg("name"), py::arg("v"))
      .def(
          "remove",
          [](const PyClass &self, const std::string &name) -> bool {
            return self.Remove(name);
          },
          py::arg("name"))
      .def(
          "search",
          [](const PyClass &self, const std::vector<float> &v, float threshold)
              -> std::string { return self.Search(v.data(), threshold); },
          py::arg("v"), py::arg("threshold"))
      .def(
          "verify",
          [](const PyClass &self, const std::string &name,
             const std::vector<float> &v, float threshold) -> bool {
            return self.Verify(name, v.data(), threshold);
          },
          py::arg("name"), py::arg("v"), py::arg("threshold"));
}

}  // namespace sherpa_onnx
