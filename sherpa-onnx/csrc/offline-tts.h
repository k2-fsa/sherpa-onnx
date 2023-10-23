// sherpa-onnx/csrc/offline-tts.h
//
// Copyright (c)  2023  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_OFFLINE_TTS_H_
#define SHERPA_ONNX_CSRC_OFFLINE_TTS_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

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

#if __ANDROID_API__ >= 9
  OfflineTts(AAssetManager *mgr, const OfflineTtsConfig &config);
#endif

  // @param text A string containing words separated by spaces
  // @param sid Speaker ID. Used only for multi-speaker models, e.g., models
  //            trained using the VCTK dataset. It is not used for
  //            single-speaker models, e.g., models trained using the ljspeech
  //            dataset.
  GeneratedAudio Generate(const std::string &text, int64_t sid = 0,
                          float speed = 1.0) const;

 private:
  std::unique_ptr<OfflineTtsImpl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_TTS_H_
