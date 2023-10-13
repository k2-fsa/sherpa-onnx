// sherpa-onnx/csrc/wave-writer.h
//
// Copyright (c)  2023  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_WAVE_WRITER_H_
#define SHERPA_ONNX_CSRC_WAVE_WRITER_H_

#include <cstdint>
#include <string>

namespace sherpa_onnx {

// Write a single channel wave file.
// Note that the input samples are in the range [-1, 1]. It will be multiplied
// by 32767 and saved in int16_t format in the wave file.
//
// @param filename Path to save the samples.
// @param sampling_rate Sample rate of the samples.
// @param samples Pointer to the samples
// @param n Number of samples
// @return Return true if the write succeeds; return false otherwise.
bool WriteWave(const std::string &filename, int32_t sampling_rate,
               const float *samples, int32_t n);

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_WAVE_WRITER_H_
