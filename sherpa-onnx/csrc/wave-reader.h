// sherpa/csrc/wave-reader.h
//
// Copyright (c)  2023  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_WAVE_READER_H_
#define SHERPA_ONNX_CSRC_WAVE_READER_H_

#include <istream>
#include <string>
#include <vector>

namespace sherpa_onnx {

/** Read a wave file with expected sample rate.

    @param filename Path to a wave file. It MUST be single channel, PCM encoded.
    @param expected_sample_rate  Expected sample rate of the wave file. If the
                               sample rate don't match, it throws an exception.
    @param is_ok On return it is true if the reading succeeded; false otherwise.

    @return Return wave samples normalized to the range [-1, 1).
 */
std::vector<float> ReadWave(const std::string &filename,
                            float expected_sample_rate, bool *is_ok);

std::vector<float> ReadWave(std::istream &is, float expected_sample_rate,
                            bool *is_ok);

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_WAVE_READER_H_
