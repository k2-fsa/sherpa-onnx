// sherpa-onnx/python/csrc/offline-speech-denoiser.h
//
// Copyright (c)  2025  Xiaomi Corporation

#ifndef SHERPA_ONNX_PYTHON_CSRC_OFFLINE_SPEECH_DENOISER_H_
#define SHERPA_ONNX_PYTHON_CSRC_OFFLINE_SPEECH_DENOISER_H_

#include "sherpa-onnx/python/csrc/sherpa-onnx.h"

namespace sherpa_onnx {

void PybindDenoisedAudio(py::module *m);

void PybindOfflineSpeechDenoiser(py::module *m);

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_PYTHON_CSRC_OFFLINE_SPEECH_DENOISER_H_
