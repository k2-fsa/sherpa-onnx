// sherpa-onnx/csrc/offline-tts.h
//
// Copyright (c)  2023  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_OFFLINE_TTS_H_
#define SHERPA_ONNX_CSRC_OFFLINE_TTS_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "sherpa-onnx/csrc/offline-tts-model-config.h"
#include "sherpa-onnx/csrc/parse-options.h"

namespace sherpa_onnx {

struct OfflineTtsConfig {
  OfflineTtsModelConfig model;

  OfflineTtsConfig() = default;
  explicit OfflineTtsConfig(const OfflineTtsModelConfig &model)
      : model(model) {}

  void Register(ParseOptions *po);
  bool Validate() const;

  std::string ToString() const;
};

struct GeneratedAudio {
  std::vector<float> samples;
  int32_t sample_rate;
};

class OfflineTtsImpl;

class OfflineTts {
 public:
  ~OfflineTts();
  explicit OfflineTts(const OfflineTtsConfig &config);
  // @param text A string containing words separated by spaces
  GeneratedAudio Generate(const std::string &text) const;

 private:
  std::unique_ptr<OfflineTtsImpl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_TTS_H_
