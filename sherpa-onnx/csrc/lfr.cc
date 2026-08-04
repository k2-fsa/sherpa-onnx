// sherpa-onnx/csrc/lfr.cc
//
// Copyright (c)  2026  Xiaomi Corporation

#include "sherpa-onnx/csrc/lfr.h"

#include <algorithm>
#include <cassert>
#include <vector>

namespace sherpa_onnx {

std::vector<float> ApplyLfr(const std::vector<float> &input,
                            int32_t input_dim, int32_t window_size,
                            int32_t window_shift) {
  assert(input_dim > 0);
  assert(window_size > 0);
  assert(window_shift > 0);
  assert(input.size() % input_dim == 0);

  if (input.empty()) {
    return {};
  }

  int32_t input_frames = static_cast<int32_t>(input.size()) / input_dim;
  int32_t output_frames = (input_frames + window_shift - 1) / window_shift;
  int32_t output_dim = input_dim * window_size;
  std::vector<float> output(output_frames * output_dim);

  float *dst = output.data();
  for (int32_t i = 0; i != output_frames; ++i) {
    int32_t first_input_frame =
        i * window_shift - (window_size - 1) / 2;
    for (int32_t j = 0; j != window_size; ++j) {
      int32_t input_frame =
          std::clamp(first_input_frame + j, 0, input_frames - 1);
      const float *src = input.data() + input_frame * input_dim;
      std::copy(src, src + input_dim, dst);
      dst += input_dim;
    }
  }

  return output;
}

}  // namespace sherpa_onnx
