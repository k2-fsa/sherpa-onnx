// sherpa-onnx/python/csrc/online-model-config.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/python/csrc/online-model-config.h"

#include <string>
#include <vector>

#include "sherpa-onnx/csrc/online-model-config.h"
#include "sherpa-onnx/csrc/online-transducer-model-config.h"
#include "sherpa-onnx/python/csrc/online-nemo-ctc-model-config.h"
#include "sherpa-onnx/python/csrc/online-paraformer-model-config.h"
#include "sherpa-onnx/python/csrc/online-transducer-model-config.h"
#include "sherpa-onnx/python/csrc/online-wenet-ctc-model-config.h"
#include "sherpa-onnx/python/csrc/online-zipformer2-ctc-model-config.h"

namespace sherpa_onnx {

void PybindOnlineModelConfig(py::module *m) {
  PybindOnlineTransducerModelConfig(m);
  PybindOnlineParaformerModelConfig(m);
  PybindOnlineWenetCtcModelConfig(m);
  PybindOnlineZipformer2CtcModelConfig(m);
  PybindOnlineNeMoCtcModelConfig(m);

  using PyClass = OnlineModelConfig;
  py::class_<PyClass>(*m, "OnlineModelConfig")
      .def(py::init<const OnlineTransducerModelConfig &,
                    const OnlineParaformerModelConfig &,
                    const OnlineWenetCtcModelConfig &,
                    const OnlineZipformer2CtcModelConfig &,
                    const OnlineNeMoCtcModelConfig &, const std::string &,
                    int32_t, int32_t, bool, const std::string &,
                    const std::string &>(),
           py::arg("transducer") = OnlineTransducerModelConfig(),
           py::arg("paraformer") = OnlineParaformerModelConfig(),
           py::arg("wenet_ctc") = OnlineWenetCtcModelConfig(),
           py::arg("zipformer2_ctc") = OnlineZipformer2CtcModelConfig(),
           py::arg("nemo_ctc") = OnlineNeMoCtcModelConfig(), py::arg("tokens"),
           py::arg("num_threads"), py::arg("warm_up") = 0,
           py::arg("debug") = false, py::arg("provider") = "cpu",
           py::arg("model_type") = "")
      .def_readwrite("transducer", &PyClass::transducer)
      .def_readwrite("paraformer", &PyClass::paraformer)
      .def_readwrite("wenet_ctc", &PyClass::wenet_ctc)
      .def_readwrite("zipformer2_ctc", &PyClass::zipformer2_ctc)
      .def_readwrite("nemo_ctc", &PyClass::nemo_ctc)
      .def_readwrite("tokens", &PyClass::tokens)
      .def_readwrite("num_threads", &PyClass::num_threads)
      .def_readwrite("debug", &PyClass::debug)
      .def_readwrite("provider", &PyClass::provider)
      .def_readwrite("model_type", &PyClass::model_type)
      .def("validate", &PyClass::Validate)
      .def("__str__", &PyClass::ToString);
}

}  // namespace sherpa_onnx
