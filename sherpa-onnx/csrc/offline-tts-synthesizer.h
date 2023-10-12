// sherpa-onnx/csrc/offline-tts-synthesizer.h
//
// Copyright (c)  2023  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_OFFLINE_TTS_SYNTHESIZER_H_
#define SHERPA_ONNX_CSRC_OFFLINE_TTS_SYNTHESIZER_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "sherpa-onnx/csrc/offline-tts-vits-model-config.h"
#include "sherpa-onnx/csrc/parse-options.h"

namespace sherpa_onnx {

struct OfflineTtsSynthesizerConfig {
  OfflineTtsVitsModelConfig vits;

  OfflineTtsSynthesizerConfig() = default;
  explicit OfflineTtsSynthesizerConfig(const OfflineTtsVitsModelConfig &vits)
      : vits(vits) {}

  void Register(ParseOptions *po);
  bool Validate() const;

  std::string ToString() const;
};

struct GeneratedAudio {
  std::vector<float> samples;
  int32_t sample_rate;
};

class OfflineTtsSynthesizerImpl;

class OfflineTtsSynthesizer {
 public:
  ~OfflineTtsSynthesizer();
  explicit OfflineTtsSynthesizer(const OfflineTtsSynthesizerConfig &config);
  // @param text A string containing words separated by spaces
  GeneratedAudio Generate(const std::string &text) const;

 private:
  std::unique_ptr<OfflineTtsSynthesizerImpl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_TTS_SYNTHESIZER_H_
