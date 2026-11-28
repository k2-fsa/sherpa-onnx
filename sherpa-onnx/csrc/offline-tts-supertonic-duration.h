// sherpa-onnx/csrc/offline-tts-supertonic-duration.h
//
// Copyright (c)  2026  linzch3

#ifndef SHERPA_ONNX_CSRC_OFFLINE_TTS_SUPERTONIC_DURATION_H_
#define SHERPA_ONNX_CSRC_OFFLINE_TTS_SUPERTONIC_DURATION_H_

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace sherpa_onnx {

// Minimum duration (in seconds) to prevent zero-length audio
constexpr float kSupertonicMinDuration = 0.1f;

// Applies `speed` to the duration (in seconds) returned by the duration
// predictor. Returns false if the duration cannot be used, i.e., it is NaN,
// infinite or not positive, either as returned by the model or after
// dividing it by `speed`.
inline bool AdjustSupertonicDuration(float speed, float *dur) {
  if (!std::isfinite(*dur) || *dur <= 0.0f) {
    return false;
  }

  if (speed != 1.0f) {
    *dur /= speed;
    // e.g., a very large speed can make it 0
    if (!std::isfinite(*dur) || *dur <= 0.0f) {
      return false;
    }

    if (*dur < kSupertonicMinDuration) {
      *dur = kSupertonicMinDuration;
    }
  }

  return true;
}

// Converts a duration (in seconds) that has passed AdjustSupertonicDuration()
// to a number of samples in the range [1, max_wav_len]. The computation is
// done in double so that a very large duration does not overflow int64_t.
inline int64_t SupertonicDurationToWavLength(float dur, int32_t sample_rate,
                                             int64_t max_wav_len) {
  double wav_len = static_cast<double>(dur) * sample_rate;
  wav_len = std::min(wav_len, static_cast<double>(max_wav_len));
  return std::max<int64_t>(static_cast<int64_t>(wav_len), 1);
}

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_TTS_SUPERTONIC_DURATION_H_
