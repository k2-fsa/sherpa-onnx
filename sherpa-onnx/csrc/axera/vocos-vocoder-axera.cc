// Copyright (c) 2026 LittleMouse
#include "sherpa-onnx/csrc/axera/vocos-vocoder-axera.h"

#include <algorithm>
#include <stdexcept>
#include <vector>

#include "kaldi-native-fbank/csrc/istft.h"
#include "sherpa-onnx/csrc/macros.h"
namespace sherpa_onnx {
VocosVocoderAxera::VocosVocoderAxera(const OfflineTtsModelConfig &config)
    : session_(config.zipvoice.vocoder, config.debug) {
  session_.CheckCount(1, 2);
  session_.CheckInput("mel", AX_ENGINE_DT_FLOAT32, {1, 100, 620});
  session_.CheckOutput("real", AX_ENGINE_DT_FLOAT32, {1, 620, 513});
  session_.CheckOutput("imag", AX_ENGINE_DT_FLOAT32, {1, 620, 513});
}
std::vector<float> VocosVocoderAxera::Run(Ort::Value mel) const {
  std::lock_guard<std::mutex> lock(mutex_);
  try {
    auto shape = mel.GetTensorTypeAndShapeInfo().GetShape();
    if (shape.size() != 3 || shape[0] != 1 || shape[1] != 100 || shape[2] < 2 ||
        shape[2] > 620) {
      throw std::runtime_error("AXERA Vocos expects [1,100,T], 2 <= T <= 620");
    }
    int32_t frames = shape[2];
    const float *data = mel.GetTensorData<float>();
    std::vector<float> padded(100 * 620, 0);
    for (int32_t c = 0; c < 100; ++c) {
      std::copy_n(data + c * frames, frames, padded.data() + c * 620);
    }
    session_.Set("mel", padded.data(), padded.size() * sizeof(float));
    session_.Run();
    knf::StftResult spectrum;
    spectrum.num_frames = frames;
    spectrum.real = session_.Get("real");
    spectrum.imag = session_.Get("imag");
    spectrum.real.resize(frames * 513);
    spectrum.imag.resize(frames * 513);
    knf::StftConfig config;
    config.n_fft = 1024;
    config.win_length = 1024;
    config.hop_length = 256;
    config.window_type = "hann";
    config.center = true;
    config.normalized = false;
    knf::IStft istft(config);
    return istft.Compute(spectrum);
  } catch (const std::exception &e) {
    SHERPA_ONNX_LOGE("AXERA Vocos failed: %s", e.what());
    return {};
  }
}
}  // namespace sherpa_onnx
