// sherpa-onnx/python/csrc/offline-model-config.cc
//
// Copyright (c)  2023 by manyeyes

#include "sherpa-onnx/python/csrc/offline-model-config.h"

#include <string>
#include <vector>

#include "sherpa-onnx/csrc/offline-model-config.h"
#include "sherpa-onnx/python/csrc/offline-moonshine-model-config.h"
#include "sherpa-onnx/python/csrc/offline-nemo-enc-dec-ctc-model-config.h"
#include "sherpa-onnx/python/csrc/offline-paraformer-model-config.h"
#include "sherpa-onnx/python/csrc/offline-sense-voice-model-config.h"
#include "sherpa-onnx/python/csrc/offline-tdnn-model-config.h"
#include "sherpa-onnx/python/csrc/offline-transducer-model-config.h"
#include "sherpa-onnx/python/csrc/offline-wenet-ctc-model-config.h"
#include "sherpa-onnx/python/csrc/offline-whisper-model-config.h"
#include "sherpa-onnx/python/csrc/offline-zipformer-ctc-model-config.h"

namespace sherpa_onnx {

void PybindOfflineModelConfig(py::module *m) {
  PybindOfflineTransducerModelConfig(m);
  PybindOfflineParaformerModelConfig(m);
  PybindOfflineNemoEncDecCtcModelConfig(m);
  PybindOfflineWhisperModelConfig(m);
  PybindOfflineTdnnModelConfig(m);
  PybindOfflineZipformerCtcModelConfig(m);
  PybindOfflineWenetCtcModelConfig(m);
  PybindOfflineSenseVoiceModelConfig(m);
  PybindOfflineMoonshineModelConfig(m);

  using PyClass = OfflineModelConfig;
  py::class_<PyClass>(*m, "OfflineModelConfig")
      .def(
          py::init<
              const OfflineTransducerModelConfig &,
              const OfflineParaformerModelConfig &,
              const OfflineNemoEncDecCtcModelConfig &,
              const OfflineWhisperModelConfig &, const OfflineTdnnModelConfig &,
              const OfflineZipformerCtcModelConfig &,
              const OfflineWenetCtcModelConfig &,
              const OfflineSenseVoiceModelConfig &,
              const OfflineMoonshineModelConfig &, const std::string &,
              const std::string &, int32_t, bool, const std::string &,
              const std::string &, const std::string &, const std::string &>(),
          py::arg("transducer") = OfflineTransducerModelConfig(),
          py::arg("paraformer") = OfflineParaformerModelConfig(),
          py::arg("nemo_ctc") = OfflineNemoEncDecCtcModelConfig(),
          py::arg("whisper") = OfflineWhisperModelConfig(),
          py::arg("tdnn") = OfflineTdnnModelConfig(),
          py::arg("zipformer_ctc") = OfflineZipformerCtcModelConfig(),
          py::arg("wenet_ctc") = OfflineWenetCtcModelConfig(),
          py::arg("sense_voice") = OfflineSenseVoiceModelConfig(),
          py::arg("moonshine") = OfflineMoonshineModelConfig(),
          py::arg("telespeech_ctc") = "", py::arg("tokens"),
          py::arg("num_threads"), py::arg("debug") = false,
          py::arg("provider") = "cpu", py::arg("model_type") = "",
          py::arg("modeling_unit") = "cjkchar", py::arg("bpe_vocab") = "")
      .def_readwrite("transducer", &PyClass::transducer)
      .def_readwrite("paraformer", &PyClass::paraformer)
      .def_readwrite("nemo_ctc", &PyClass::nemo_ctc)
      .def_readwrite("whisper", &PyClass::whisper)
      .def_readwrite("tdnn", &PyClass::tdnn)
      .def_readwrite("zipformer_ctc", &PyClass::zipformer_ctc)
      .def_readwrite("wenet_ctc", &PyClass::wenet_ctc)
      .def_readwrite("sense_voice", &PyClass::sense_voice)
      .def_readwrite("moonshine", &PyClass::moonshine)
      .def_readwrite("telespeech_ctc", &PyClass::telespeech_ctc)
      .def_readwrite("tokens", &PyClass::tokens)
      .def_readwrite("num_threads", &PyClass::num_threads)
      .def_readwrite("debug", &PyClass::debug)
      .def_readwrite("provider", &PyClass::provider)
      .def_readwrite("model_type", &PyClass::model_type)
      .def_readwrite("modeling_unit", &PyClass::modeling_unit)
      .def_readwrite("bpe_vocab", &PyClass::bpe_vocab)
      .def("validate", &PyClass::Validate)
      .def("__str__", &PyClass::ToString);
}

}  // namespace sherpa_onnx
