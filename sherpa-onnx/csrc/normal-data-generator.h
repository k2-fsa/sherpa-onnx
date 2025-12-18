// sherpa-onnx/csrc/normal-data-generator.h
//
// Copyright      2025  Xiaomi Corporation

// Written by ChatGPT

#include <cstddef>

namespace sherpa_onnx {

class NormalDataGenerator {
 public:
  NormalDataGenerator(float mean = 0.0f, float stddev = 1.0f);

  // Fill pre-allocated memory
  void Fill(float *data, std::size_t size) const;

 private:
  float mean_;
  float stddev_;
};

}  // namespace sherpa_onnx
