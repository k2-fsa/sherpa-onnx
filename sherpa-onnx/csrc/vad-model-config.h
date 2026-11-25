// sherpa-onnx/csrc/vad-model-config.h
//
// Copyright (c)  2023  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_VAD_MODEL_CONFIG_H_
#define SHERPA_ONNX_CSRC_VAD_MODEL_CONFIG_H_

#include <string>

#include "sherpa-onnx/csrc/nomo-pvad-model-config.h"
#include "sherpa-onnx/csrc/parse-options.h"
#include "sherpa-onnx/csrc/silero-vad-model-config.h"
#include "sherpa-onnx/csrc/ten-vad-model-config.h"

namespace sherpa_onnx {

struct VadModelConfig {
  SileroVadModelConfig silero_vad;
  TenVadModelConfig ten_vad;

  // A target-speaker VAD: it detects whether the enrolled speaker is
  // talking instead of whether anybody is talking.
  NomoPvadModelConfig nomo_pvad;

  int32_t sample_rate = 16000;
  int32_t num_threads = 1;
  std::string provider = "cpu";

  // true to show debug information when loading models
  bool debug = false;

  VadModelConfig() = default;

  VadModelConfig(const SileroVadModelConfig &silero_vad,
                 const TenVadModelConfig &ten_vad,
                 const NomoPvadModelConfig &nomo_pvad, int32_t sample_rate,
                 int32_t num_threads, const std::string &provider, bool debug)
      : silero_vad(silero_vad),
        ten_vad(ten_vad),
        nomo_pvad(nomo_pvad),
        sample_rate(sample_rate),
        num_threads(num_threads),
        provider(provider),
        debug(debug) {}

  // Kept for backward compatibility. It is equivalent to the constructor
  // above with an empty NomoPvadModelConfig.
  VadModelConfig(const SileroVadModelConfig &silero_vad,
                 const TenVadModelConfig &ten_vad, int32_t sample_rate,
                 int32_t num_threads, const std::string &provider, bool debug)
      : VadModelConfig(silero_vad, ten_vad, NomoPvadModelConfig{}, sample_rate,
                       num_threads, provider, debug) {}

  void Register(ParseOptions *po);
  bool Validate() const;

  // Like Validate(), for model files given by a resource manager (Android
  // asset manager / OHOS native resource manager): model paths in
  // `silero_vad`, `ten_vad` and `nomo_pvad` are relative to the resource
  // root.
  template <typename Manager>
  bool Validate(Manager *mgr) const;

  std::string ToString() const;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_VAD_MODEL_CONFIG_H_
