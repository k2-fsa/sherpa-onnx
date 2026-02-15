// sherpa-onnx/csrc/normal-data-generator.cc
//
// Copyright      2025  Xiaomi Corporation

#include "sherpa-onnx/csrc/normal-data-generator.h"

#include <random>
#include <thread>

namespace sherpa_onnx {

// Helper type hidden in translation unit
namespace {
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
}  // namespace

NormalDataGenerator::NormalDataGenerator(float mean /* = 0.0f */,
                                         float stddev /* = 1.0f */)
    : mean_(mean), stddev_(stddev), seed_(-1) {}

NormalDataGenerator::NormalDataGenerator(float mean, float stddev, int32_t seed)
    : mean_(mean), stddev_(stddev), seed_(seed) {
  if (seed_ >= 0) {
    rng_.seed(static_cast<unsigned>(seed_));
  }
}

void NormalDataGenerator::Fill(float *data, std::size_t size) const {
  if (seed_ >= 0) {
    // Deterministic mode: use instance-level RNG
    std::normal_distribution<float> dist(mean_, stddev_);
    for (std::size_t i = 0; i < size; ++i) {
      data[i] = dist(rng_);
    }
  } else {
    // Original behavior: thread-local random device
    static thread_local RNGHolder holder;

    holder.dist.param(
        std::normal_distribution<float>::param_type(mean_, stddev_));

    for (std::size_t i = 0; i < size; ++i) {
      data[i] = holder.dist(holder.rng);
    }
  }
}

}  // namespace sherpa_onnx
