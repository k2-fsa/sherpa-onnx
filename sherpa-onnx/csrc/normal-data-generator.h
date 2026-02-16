// sherpa-onnx/csrc/normal-data-generator.h
//
// Copyright      2025  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_NORMAL_DATA_GENERATOR_H_
#define SHERPA_ONNX_CSRC_NORMAL_DATA_GENERATOR_H_

#include <cstddef>
#include <cstdint>
#include <random>

namespace sherpa_onnx {

class NormalDataGenerator {
 public:
  explicit NormalDataGenerator(float mean = 0.0f, float stddev = 1.0f,
                               int32_t seed = -1);

  // Fill pre-allocated memory
  void Fill(float *data, std::size_t size) const;

 private:
  float mean_;
  float stddev_;
  int32_t seed_ = -1;         // -1 = use thread-local random device (default)
  mutable std::mt19937 rng_;  // used if seed_ >= 0
};

}  // namespace sherpa_onnx
#endif  // SHERPA_ONNX_CSRC_NORMAL_DATA_GENERATOR_H_
