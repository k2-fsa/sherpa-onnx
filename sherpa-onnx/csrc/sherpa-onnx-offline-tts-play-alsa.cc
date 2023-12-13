// sherpa-onnx/csrc/sherpa-onnx-tts-play-alsa.cc
//
// Copyright (c)  2022-2023  Xiaomi Corporation

// see https://www.alsa-project.org/alsa-doc/alsa-lib/group___p_c_m.html
// https://www.alsa-project.org/alsa-doc/alsa-lib/group___p_c_m___h_w___params.html
// https://www.alsa-project.org/alsa-doc/alsa-lib/group___p_c_m.html

#include "alsa/asoundlib.h"
#include <vector>
#include <cstdint>
#include "sherpa-onnx/csrc/wave-reader.h"
#include "sherpa-onnx/csrc/alsa-play.h"

int main() {
  snd_pcm_t *handle;
  const char* device_name="default";

  bool is_ok;

  int32_t sample_rate;
  const std::vector<float> float_samples =
        sherpa_onnx::ReadWave("test.wav", &sample_rate, &is_ok);

  sherpa_onnx::AlsaPlay alsa(device_name, sample_rate);
  alsa.Play(float_samples);
  alsa.Drain();

  return 0 ;
}
