// sherpa-onnx/csrc/normal-data-generator.cc
//
// Copyright      2025  Xiaomi Corporation

// Written by ChatGPT

#include "sherpa-onnx/csrc/normal-data-generator.h"

#include <random>
#include <thread>

namespace sherpa_onnx {

// Helper type hidden in translation unit
struct RNGHolder {
  std::mt19937 rng;
  std::normal_distribution<float> dist;

  RNGHolder()
      : rng([] {
          std::random_device rd;
          std::seed_seq seq{rd(),
                            static_cast<unsigned>(std::hash<std::thread::id>{}(
                                std::this_thread::get_id()))};
          return std::mt19937(seq);
        }()),
        dist() {}
};

NormalDataGenerator::NormalDataGenerator(float mean /* = 0.0f */,
                                         float stddev /* = 1.0f */)
    : mean_(mean), stddev_(stddev) {}

void NormalDataGenerator::Fill(float *data, std::size_t size) const {
  // One RNGHolder per thread
  static thread_local RNGHolder holder;

  holder.dist.param(
      std::normal_distribution<float>::param_type(mean_, stddev_));

  for (std::size_t i = 0; i < size; ++i) {
    data[i] = holder.dist(holder.rng);
  }
}

}  // namespace sherpa_onnx
