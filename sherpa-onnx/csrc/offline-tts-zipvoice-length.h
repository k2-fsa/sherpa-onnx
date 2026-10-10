// Copyright (c) 2026 LittleMouse
#ifndef SHERPA_ONNX_CSRC_OFFLINE_TTS_ZIPVOICE_LENGTH_H_
#define SHERPA_ONNX_CSRC_OFFLINE_TTS_ZIPVOICE_LENGTH_H_
#include <cmath>
#include <cstdint>
namespace sherpa_onnx {
// Returns zero instead of silently clamping a static model's duration.
inline int32_t ZipvoiceStaticFeatureLength(int32_t prompt_tokens,
                                           int32_t text_tokens,
                                           int32_t prompt_frames, float speed,
                                           int32_t max_tokens,
                                           int32_t max_frames,
                                           int32_t max_generated_frames) {
  if (prompt_tokens <= 0 || text_tokens <= 0 || prompt_frames <= 0 ||
      !std::isfinite(speed) || speed <= 0 || prompt_tokens >= max_tokens ||
      text_tokens >= max_tokens - prompt_tokens || prompt_frames >= max_frames)
    return 0;
  double frames = std::ceil(static_cast<double>(prompt_frames) / prompt_tokens *
                            (prompt_tokens + text_tokens) / speed);
  if (frames < prompt_tokens + text_tokens || frames <= prompt_frames + 1 ||
      frames > max_frames || frames - prompt_frames > max_generated_frames)
    return 0;
  return static_cast<int32_t>(frames);
}
}  // namespace sherpa_onnx
#endif  // SHERPA_ONNX_CSRC_OFFLINE_TTS_ZIPVOICE_LENGTH_H_
