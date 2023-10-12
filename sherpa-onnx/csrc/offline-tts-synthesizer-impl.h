// sherpa-onnx/csrc/offline-tts-synthesizer-impl.h
//
// Copyright (c)  2023  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_OFFLINE_TTS_SYNTHESIZER_IMPL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_TTS_SYNTHESIZER_IMPL_H_

#include <memory>
#include <string>

#include "sherpa-onnx/csrc/offline-tts-synthesizer.h"

namespace sherpa_onnx {

class OfflineTtsSynthesizerImpl {
 public:
  virtual ~OfflineTtsSynthesizerImpl() = default;

  static std::unique_ptr<OfflineTtsSynthesizerImpl> Create(
      const OfflineTtsSynthesizerConfig &config);

  virtual GeneratedAudio Generate(const std::string &text) const = 0;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_TTS_SYNTHESIZER_IMPL_H_
