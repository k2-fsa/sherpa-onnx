// sherpa-onnx/csrc/lfr.cc
//
// Copyright (c)  2026  zhifu gao

#include "sherpa-onnx/csrc/lfr.h"

#include <algorithm>
#include <limits>
#include <vector>

#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {

namespace {

size_t GetOutputDim(int32_t input_dim, int32_t window_size,
                    const char *caller) {
  const size_t input_dim_size = static_cast<size_t>(input_dim);
  const size_t window_size_size = static_cast<size_t>(window_size);
  const size_t max_output_dim =
      static_cast<size_t>(std::numeric_limits<int32_t>::max());
  if (window_size_size > max_output_dim / input_dim_size) {
    SHERPA_ONNX_LOGE(
        "%s: output dimension %d * %d exceeds int32 capacity", caller,
        input_dim, window_size);
    SHERPA_ONNX_EXIT(-1);
  }

  return input_dim_size * window_size_size;
}

size_t GetOutputSize(size_t output_frames, size_t output_dim,
                     const char *caller) {
  const std::vector<float> empty;
  if (output_frames > empty.max_size() / output_dim) {
    SHERPA_ONNX_LOGE(
        "%s: output with %zu frames and dimension %zu exceeds vector "
        "capacity",
        caller, output_frames, output_dim);
    SHERPA_ONNX_EXIT(-1);
  }

  return output_frames * output_dim;
}

}  // namespace

std::vector<float> ApplyLfr(const std::vector<float> &input,
                            int32_t input_dim, int32_t window_size,
                            int32_t window_shift) {
  if (input_dim <= 0) {
    SHERPA_ONNX_LOGE("ApplyLfr: input_dim must be positive. Given: %d",
                     input_dim);
    SHERPA_ONNX_EXIT(-1);
  }

  if (window_size <= 0) {
    SHERPA_ONNX_LOGE("ApplyLfr: window_size must be positive. Given: %d",
                     window_size);
    SHERPA_ONNX_EXIT(-1);
  }

  if (window_shift <= 0) {
    SHERPA_ONNX_LOGE("ApplyLfr: window_shift must be positive. Given: %d",
                     window_shift);
    SHERPA_ONNX_EXIT(-1);
  }

  if (input.size() % static_cast<size_t>(input_dim) != 0) {
    SHERPA_ONNX_LOGE(
        "ApplyLfr: input size %zu is not divisible by input_dim %d",
        input.size(), input_dim);
    SHERPA_ONNX_EXIT(-1);
  }

  if (input.empty()) {
    return {};
  }

  const size_t input_dim_size = static_cast<size_t>(input_dim);
  const size_t window_size_size = static_cast<size_t>(window_size);
  const size_t window_shift_size = static_cast<size_t>(window_shift);
  const size_t input_frames = input.size() / input_dim_size;
  if (input_frames >
      static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
    SHERPA_ONNX_LOGE("ApplyLfr: input frame count %zu exceeds int32 capacity",
                     input_frames);
    SHERPA_ONNX_EXIT(-1);
  }

  const size_t output_frames =
      1 + (input_frames - 1) / window_shift_size;
  const size_t output_dim =
      GetOutputDim(input_dim, window_size, "ApplyLfr");
  std::vector<float> output(
      GetOutputSize(output_frames, output_dim, "ApplyLfr"));

  float *dst = output.data();
  const size_t left_context = (window_size_size - 1) / 2;
  for (size_t i = 0; i != output_frames; ++i) {
    const size_t center_frame = i * window_shift_size;
    const size_t left_padding =
        center_frame < left_context ? left_context - center_frame : 0;
    const size_t first_input_frame =
        center_frame < left_context ? 0 : center_frame - left_context;
    const size_t max_offset = input_frames - 1 - first_input_frame;

    for (size_t j = 0; j != window_size_size; ++j) {
      size_t input_frame = 0;
      if (j >= left_padding) {
        const size_t offset = j - left_padding;
        input_frame = offset > max_offset
                          ? input_frames - 1
                          : first_input_frame + offset;
      }

      const float *src = input.data() + input_frame * input_dim_size;
      std::copy(src, src + input_dim_size, dst);
      dst += input_dim_size;
    }
  }

  return output;
}

std::vector<float> ApplyLfrForFixedShape(
    const std::vector<float> &input, int32_t input_dim, int32_t window_size,
    int32_t window_shift, int32_t output_frames) {
  if (output_frames <= 0) {
    SHERPA_ONNX_LOGE(
        "ApplyLfrForFixedShape: output_frames must be positive. Given: %d",
        output_frames);
    SHERPA_ONNX_EXIT(-1);
  }

  std::vector<float> output =
      ApplyLfr(input, input_dim, window_size, window_shift);
  if (output.empty()) {
    return {};
  }

  const size_t output_dim =
      GetOutputDim(input_dim, window_size, "ApplyLfrForFixedShape");
  const size_t output_frames_size = static_cast<size_t>(output_frames);
  const size_t target_size = GetOutputSize(
      output_frames_size, output_dim, "ApplyLfrForFixedShape");
  const size_t actual_output_frames = output.size() / output_dim;
  if (actual_output_frames > output_frames_size) {
    SHERPA_ONNX_LOGE(
        "Number of input frames %zu is too large. Truncate it to %d frames.",
        actual_output_frames, output_frames);
    SHERPA_ONNX_LOGE(
        "Recognition result may be truncated/incomplete. Please select a "
        "model accepting longer audios.");
  }

  output.resize(target_size, 0.0f);
  return output;
}

}  // namespace sherpa_onnx
