// sherpa-onnx/csrc/offline-tts-vits-impl.h
//
// Copyright (c)  2023  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_OFFLINE_TTS_VITS_IMPL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_TTS_VITS_IMPL_H_

#include <fstream>
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

  GeneratedAudio Generate(const std::string &text) const override {
    SHERPA_ONNX_LOGE("txt: %s", text.c_str());
    auto memory_info =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

    std::vector<int64_t> x = {
        0, 54,  0, 157, 0, 102, 0, 54, 0, 51,  0, 158, 0, 156, 0, 72,
        0, 56,  0, 83,  0, 3,   0, 16, 0, 157, 0, 43,  0, 135, 0, 85,
        0, 16,  0, 55,  0, 156, 0, 57, 0, 135, 0, 61,  0, 62,  0, 16,
        0, 44,  0, 52,  0, 156, 0, 63, 0, 158, 0, 125, 0, 102, 0, 48,
        0, 83,  0, 54,  0, 16,  0, 72, 0, 56,  0, 46,  0, 16,  0, 54,
        0, 156, 0, 138, 0, 64,  0, 54, 0, 51,  0, 16,  0, 70,  0, 61,
        0, 156, 0, 102, 0, 61,  0, 62, 0, 83,  0, 56,  0, 62,  0};
    std::array<int64_t, 2> x_shape = {1, static_cast<int32_t>(x.size())};
    Ort::Value x_tensor = Ort::Value::CreateTensor(
        memory_info, x.data(), x.size(), x_shape.data(), x_shape.size());
    Ort::Value audio = model_->Run(std::move(x_tensor));

    std::vector<int64_t> audio_shape =
        audio.GetTensorTypeAndShapeInfo().GetShape();

    SHERPA_ONNX_LOGE("%d, %d", int(audio_shape.size()), int(audio_shape[2]));
    const float *p = audio.GetTensorData<float>();
    std::ofstream os("t.pcm", std::ios::binary);
    os.write(reinterpret_cast<const char *>(p), sizeof(float) * audio_shape[2]);

    // sox -t raw -r 22050 -b 32 -e floating-point -c 1 ./t.pcm ./t.wav

    return {};
  }

 private:
  std::unique_ptr<OfflineTtsVitsModel> model_;
};

}  // namespace sherpa_onnx
#endif  // SHERPA_ONNX_CSRC_OFFLINE_TTS_VITS_IMPL_H_
