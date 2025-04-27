// sherpa-onnx/python/csrc/homophone-replacer.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa-onnx/python/csrc/homophone-replacer.h"

#include <string>

#include "sherpa-onnx/csrc/homophone-replacer.h"

namespace sherpa_onnx {

void PybindHomophoneReplacer(py::module *m) {
  using PyClass = HomophoneReplacerConfig;
  py::class_<PyClass>(*m, "HomophoneReplacerConfig")
      .def(py::init<>())
      .def(py::init<const std::string &, const std::string &,
                    const std::string &, bool>(),
           py::arg("dict_dir"), py::arg("lexicon"), py::arg("rule_fsts"),
           py::arg("debug") = false)
      .def_readwrite("dict_dir", &PyClass::dict_dir)
      .def_readwrite("lexicon", &PyClass::lexicon)
      .def_readwrite("rule_fsts", &PyClass::rule_fsts)
      .def_readwrite("debug", &PyClass::debug)
      .def("__str__", &PyClass::ToString);
}

}  // namespace sherpa_onnx
