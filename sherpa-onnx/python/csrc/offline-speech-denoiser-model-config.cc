// sherpa-onnx/python/csrc/offline-speech-denoiser-model-config.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa-onnx/python/csrc/offline-speech-denoiser-model-config.h"

#include <string>

#include "sherpa-onnx/csrc/offline-speech-denoiser-model-config.h"
#include "sherpa-onnx/python/csrc/offline-speech-denoiser-dpdfnet-model-config.h"
#include "sherpa-onnx/python/csrc/offline-speech-denoiser-gtcrn-model-config.h"
#include "sherpa-onnx/python/csrc/offline-speech-denoiser-voxlock-model-config.h"

namespace sherpa_onnx {

void PybindOfflineSpeechDenoiserModelConfig(py::module *m) {
  PybindOfflineSpeechDenoiserDpdfNetModelConfig(m);
  PybindOfflineSpeechDenoiserGtcrnModelConfig(m);
  PybindOfflineSpeechDenoiserVoxLockModelConfig(m);

  using PyClass = OfflineSpeechDenoiserModelConfig;
  py::class_<PyClass>(*m, "OfflineSpeechDenoiserModelConfig")
      .def(py::init<>())
      .def(py::init([](const OfflineSpeechDenoiserGtcrnModelConfig &gtcrn,
                       const OfflineSpeechDenoiserDpdfNetModelConfig &dpdfnet,
                       const OfflineSpeechDenoiserVoxLockModelConfig &voxlock,
                       int32_t num_threads, bool debug,
                       const std::string &provider) {
             PyClass ans;
             ans.gtcrn = gtcrn;
             ans.dpdfnet = dpdfnet;
             ans.voxlock = voxlock;
             ans.num_threads = num_threads;
             ans.debug = debug;
             ans.provider = provider;
             return ans;
           }),
           py::arg("gtcrn") = OfflineSpeechDenoiserGtcrnModelConfig{},
           py::arg("dpdfnet") = OfflineSpeechDenoiserDpdfNetModelConfig{},
           py::arg("voxlock") = OfflineSpeechDenoiserVoxLockModelConfig{},
           py::arg("num_threads") = 1, py::arg("debug") = false,
           py::arg("provider") = "cpu")
      .def_readwrite("gtcrn", &PyClass::gtcrn)
      .def_readwrite("dpdfnet", &PyClass::dpdfnet)
      .def_readwrite("voxlock", &PyClass::voxlock)
      .def_readwrite("num_threads", &PyClass::num_threads)
      .def_readwrite("debug", &PyClass::debug)
      .def_readwrite("provider", &PyClass::provider)
      .def("validate", &PyClass::Validate)
      .def("__str__", &PyClass::ToString);
}

}  // namespace sherpa_onnx
