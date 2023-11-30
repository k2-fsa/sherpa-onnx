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
  // If not empty, it contains a list of rule FST filenames.
  // Filenames are separated by a comma.
  // Example value: rule1.fst,rule2,fst,rule3.fst
  //
  // If there are multiple rules, they are applied from left to right.
  std::string rule_fsts;

  // Maximum number of sentences that we process at a time.
  // This is to avoid OOM for very long input text.
  // If you set it to -1, then we process all sentences in a single batch.
  int32_t max_num_sentences = 2;

  OfflineTtsConfig() = default;
  OfflineTtsConfig(const OfflineTtsModelConfig &model,
                   const std::string &rule_fsts, int32_t max_num_sentences)
      : model(model),
        rule_fsts(rule_fsts),
        max_num_sentences(max_num_sentences) {}

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
