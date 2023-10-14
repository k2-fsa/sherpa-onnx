// sherpa-onnx/python/csrc/sherpa-onnx.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/python/csrc/sherpa-onnx.h"

#include "sherpa-onnx/python/csrc/circular-buffer.h"
#include "sherpa-onnx/python/csrc/display.h"
#include "sherpa-onnx/python/csrc/endpoint.h"
#include "sherpa-onnx/python/csrc/features.h"
#include "sherpa-onnx/python/csrc/offline-ctc-fst-decoder-config.h"
#include "sherpa-onnx/python/csrc/offline-lm-config.h"
#include "sherpa-onnx/python/csrc/offline-model-config.h"
#include "sherpa-onnx/python/csrc/offline-recognizer.h"
#include "sherpa-onnx/python/csrc/offline-stream.h"
#include "sherpa-onnx/python/csrc/offline-tts.h"
#include "sherpa-onnx/python/csrc/online-lm-config.h"
#include "sherpa-onnx/python/csrc/online-model-config.h"
#include "sherpa-onnx/python/csrc/online-recognizer.h"
#include "sherpa-onnx/python/csrc/online-stream.h"
#include "sherpa-onnx/python/csrc/vad-model-config.h"
#include "sherpa-onnx/python/csrc/vad-model.h"
#include "sherpa-onnx/python/csrc/voice-activity-detector.h"

namespace sherpa_onnx {

PYBIND11_MODULE(_sherpa_onnx, m) {
  m.doc() = "pybind11 binding of sherpa-onnx";

  PybindFeatures(&m);
  PybindOnlineModelConfig(&m);
  PybindOnlineLMConfig(&m);
  PybindOnlineStream(&m);
  PybindEndpoint(&m);
  PybindOnlineRecognizer(&m);

  PybindDisplay(&m);

  PybindOfflineStream(&m);
  PybindOfflineLMConfig(&m);
  PybindOfflineModelConfig(&m);
  PybindOfflineCtcFstDecoderConfig(&m);
  PybindOfflineRecognizer(&m);

  PybindVadModelConfig(&m);
  PybindVadModel(&m);
  PybindCircularBuffer(&m);
  PybindVoiceActivityDetector(&m);

  PybindOfflineTts(&m);
}

}  // namespace sherpa_onnx
