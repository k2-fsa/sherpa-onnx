// sherpa-onnx/csrc/offline-tts-vits-impl.h
//
// Copyright (c)  2023  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_OFFLINE_TTS_VITS_IMPL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_TTS_VITS_IMPL_H_

#include <string>

#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/offline-tts-impl.h"
#include "sherpa-onnx/csrc/offline-tts-vits-model.h"

namespace sherpa_onnx {

class OfflineTtsVitsImpl : public OfflineTtsImpl {
 public:
  explicit OfflineTtsVitsImpl(const OfflineTtsConfig &config)
      : model_(std::make_unique<OfflineTtsVitsModel>(config.model)) {
    SHERPA_ONNX_LOGE("config: %s\n", config.ToString().c_str());
  }

  GeneratedAudio Generate(const std::string &text) const override { return {}; }

 private:
  std::unique_ptr<OfflineTtsVitsModel> model_;
};

}  // namespace sherpa_onnx
#endif  // SHERPA_ONNX_CSRC_OFFLINE_TTS_VITS_IMPL_H_
