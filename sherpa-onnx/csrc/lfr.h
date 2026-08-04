// sherpa-onnx/csrc/lfr.h
//
// Copyright (c)  2026  zhifu gao

#ifndef SHERPA_ONNX_CSRC_LFR_H_
#define SHERPA_ONNX_CSRC_LFR_H_

#include <cstdint>
#include <vector>

namespace sherpa_onnx {

/** Stack input frames for low frame rate (LFR) processing.
 *
 * The first frame is repeated on the left and the last frame is repeated on
 * the right so every output window is complete. The result contains
 * ceil(num_frames / window_shift) output frames.
 *
 * @param input Flattened input with shape (num_frames, input_dim).
 * @param input_dim Feature dimension of one input frame.
 * @param window_size Number of input frames stacked into each output frame.
 * @param window_shift Number of input frames between adjacent output frames.
 *
 * @return Flattened output with shape
 *         (num_output_frames, input_dim * window_size).
 */
std::vector<float> ApplyLfr(const std::vector<float> &input,
                            int32_t input_dim, int32_t window_size,
                            int32_t window_shift);

/** Apply LFR and resize it for a model with a fixed frame dimension.
 *
 * The LFR result is truncated when it exceeds output_frames and zero-padded
 * when it is shorter. Empty input remains empty so callers can skip inference.
 *
 * @param input Flattened input with shape (num_frames, input_dim).
 * @param input_dim Feature dimension of one input frame.
 * @param window_size Number of input frames stacked into each output frame.
 * @param window_shift Number of input frames between adjacent output frames.
 * @param output_frames Required output frame dimension of the model.
 *
 * @return Flattened output with shape
 *         (output_frames, input_dim * window_size), or empty for empty input.
 */
std::vector<float> ApplyLfrForFixedShape(
    const std::vector<float> &input, int32_t input_dim, int32_t window_size,
    int32_t window_shift, int32_t output_frames);

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_LFR_H_
