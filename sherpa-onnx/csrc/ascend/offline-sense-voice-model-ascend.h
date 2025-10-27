// sherpa-onnx/csrc/ascend/offline-sense-voice-model-ascend.h
//
// Copyright (c)  2025  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_ASCEND_OFFLINE_SENSE_VOICE_MODEL_ASCEND_H_
#define SHERPA_ONNX_CSRC_ASCEND_OFFLINE_SENSE_VOICE_MODEL_ASCEND_H_

#include <memory>

namespace sherpa_onnx {

class OfflineSenseVoiceModelAscend {
 public:
  OfflineSenseVoiceModelAscend();
  ~OfflineSenseVoiceModelAscend();

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_ASCEND_OFFLINE_SENSE_VOICE_MODEL_ASCEND_H_
