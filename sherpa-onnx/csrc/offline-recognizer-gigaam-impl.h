// sherpa-onnx/csrc/offline-recognizer-gigaam-impl.h
//
// Copyright (c)  2026  sherpa-onnx contributors

#ifndef SHERPA_ONNX_CSRC_OFFLINE_RECOGNIZER_GIGAAM_IMPL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_RECOGNIZER_GIGAAM_IMPL_H_

#include <memory>

namespace sherpa_onnx {

class OfflineRecognizerImpl;
struct OfflineRecognizerConfig;

std::unique_ptr<OfflineRecognizerImpl> CreateOfflineRecognizerGigaAM(
    const OfflineRecognizerConfig &config);

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_RECOGNIZER_GIGAAM_IMPL_H_
