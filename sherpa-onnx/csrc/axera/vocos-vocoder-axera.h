// Copyright (c) 2026 LittleMouse
#ifndef SHERPA_ONNX_CSRC_AXERA_VOCOS_VOCODER_AXERA_H_
#define SHERPA_ONNX_CSRC_AXERA_VOCOS_VOCODER_AXERA_H_
#include <mutex>
#include <vector>

#include "sherpa-onnx/csrc/axera/tts-session.h"
#include "sherpa-onnx/csrc/vocoder.h"
namespace sherpa_onnx {
class VocosVocoderAxera : public Vocoder {
 public:
  explicit VocosVocoderAxera(const OfflineTtsModelConfig &config);
  std::vector<float> Run(Ort::Value mel) const override;

 private:
  mutable std::mutex mutex_;
  mutable AxeraTtsSession session_;
};
}  // namespace sherpa_onnx
#endif  // SHERPA_ONNX_CSRC_AXERA_VOCOS_VOCODER_AXERA_H_
