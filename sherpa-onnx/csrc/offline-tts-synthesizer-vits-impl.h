// sherpa-onnx/csrc/offline-tts-synthesizer-vits-impl.h
//
// Copyright (c)  2023  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_OFFLINE_TTS_SYNTHESIZER_VITS_IMPL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_TTS_SYNTHESIZER_VITS_IMPL_H_

#include <string>

#include "sherpa-onnx/csrc/offline-tts-synthesizer-impl.h"

namespace sherpa_onnx {

class OfflineTtsSynthesizerVitsImpl : public OfflineTtsSynthesizerImpl {
 public:
  explicit OfflineTtsSynthesizerVitsImpl(
      const OfflineTtsSynthesizerConfig &config) {}

  GeneratedAudio Generate(const std::string &text) const override { return {}; }

 private:
};

}  // namespace sherpa_onnx
#endif  // SHERPA_ONNX_CSRC_OFFLINE_TTS_SYNTHESIZER_VITS_IMPL_H_
