// sherpa-onnx/csrc/session.h
//
// Copyright (c)  2023  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_SESSION_H_
#define SHERPA_ONNX_CSRC_SESSION_H_

#include "onnxruntime_cxx_api.h"  // NOLINT
#include "sherpa-onnx/csrc/offline-lm-config.h"
#include "sherpa-onnx/csrc/offline-model-config.h"
#include "sherpa-onnx/csrc/offline-tts-model-config.h"
#include "sherpa-onnx/csrc/online-lm-config.h"
#include "sherpa-onnx/csrc/online-model-config.h"
#include "sherpa-onnx/csrc/speaker-embedding-extractor.h"
#include "sherpa-onnx/csrc/vad-model-config.h"

namespace sherpa_onnx {

Ort::SessionOptions GetSessionOptions(const OnlineModelConfig &config);

Ort::SessionOptions GetSessionOptions(const OfflineModelConfig &config);

Ort::SessionOptions GetSessionOptions(const OfflineLMConfig &config);

Ort::SessionOptions GetSessionOptions(const OnlineLMConfig &config);

Ort::SessionOptions GetSessionOptions(const VadModelConfig &config);

Ort::SessionOptions GetSessionOptions(const OfflineTtsModelConfig &config);

Ort::SessionOptions GetSessionOptions(
    const SpeakerEmbeddingExtractorConfig &config);
}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_SESSION_H_
