// sherpa-onnx/csrc/onnx-env.cc
//
// Copyright (c)  2026  Xiaomi Corporation

#include "sherpa-onnx/csrc/onnx-env.h"

#include <memory>
#include <mutex>
#include <thread>  // NOLINT

namespace sherpa_onnx {

Ort::Env &GetGlobalOrtEnv(int32_t num_threads) {
  static std::once_flag init_flag;
  static std::unique_ptr<Ort::Env> env;
  std::call_once(init_flag, [num_threads] {
    int n = num_threads;
    if (n <= 0) {
      n = static_cast<int>(std::thread::hardware_concurrency());
      if (n <= 0) n = 1;
      if (n > 8) n = 8;
    }
    Ort::ThreadingOptions threading_options;
    threading_options.SetGlobalIntraOpNumThreads(n);
    // Keep inter-op at 1: virtually all ORT models in this repo are dominated
    // by intra-op parallelism, and inter_op > 1 just steals workers from the
    // pthread pool, hurting throughput at higher num_threads.
    threading_options.SetGlobalInterOpNumThreads(1);
    env = std::make_unique<Ort::Env>(threading_options, ORT_LOGGING_LEVEL_ERROR,
                                     "sherpa-onnx");
  });
  return *env;
}

}  // namespace sherpa_onnx
