// sherpa-onnx/csrc/session.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/session.h"

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/provider.h"
#if defined(__APPLE__)
#include "coreml_provider_factory.h"  // NOLINT
#endif

namespace sherpa_onnx {

static Ort::SessionOptions GetSessionOptionsImpl(int32_t num_threads,
                                                 std::string provider_str) {
  Provider p = StringToProvider(std::move(provider_str));

  Ort::SessionOptions sess_opts;
  sess_opts.SetIntraOpNumThreads(num_threads);
  sess_opts.SetInterOpNumThreads(num_threads);

  // Other possible options
  // sess_opts.SetGraphOptimizationLevel(ORT_ENABLE_EXTENDED);
  // sess_opts.SetLogSeverityLevel(ORT_LOGGING_LEVEL_VERBOSE);
  // sess_opts.EnableProfiling("profile");

  switch (p) {
    case Provider::kCPU:
      break;  // nothing to do for the CPU provider
    case Provider::kCUDA: {
      std::vector<std::string> available_providers =
          Ort::GetAvailableProviders();
      if (std::find(available_providers.begin(), available_providers.end(),
                    "CUDAExecutionProvider") != available_providers.end()) {
        // The CUDA provider is available, proceed with setting the options
        OrtCUDAProviderOptions options;
        options.device_id = 0;
        // Default OrtCudnnConvAlgoSearchExhaustive is extremely slow
        options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchHeuristic;
        // set more options on need
        sess_opts.AppendExecutionProvider_CUDA(options);
      } else {
        SHERPA_ONNX_LOGE(
            "Please compile with -DSHERPA_ONNX_ENABLE_GPU=ON. Fallback to "
            "cpu!");
      }
      break;
    }
    case Provider::kCoreML: {
#if defined(__APPLE__)
      uint32_t coreml_flags = 0;
      (void)OrtSessionOptionsAppendExecutionProvider_CoreML(sess_opts,
                                                            coreml_flags);
#else
      SHERPA_ONNX_LOGE("CoreML is for Apple only. Fallback to cpu!");
#endif
      break;
    }
  }

  return sess_opts;
}

Ort::SessionOptions GetSessionOptions(const OnlineModelConfig &config) {
  return GetSessionOptionsImpl(config.num_threads, config.provider);
}

Ort::SessionOptions GetSessionOptions(const OfflineModelConfig &config) {
  return GetSessionOptionsImpl(config.num_threads, config.provider);
}

Ort::SessionOptions GetSessionOptions(const OfflineLMConfig &config) {
  return GetSessionOptionsImpl(config.lm_num_threads, config.lm_provider);
}

Ort::SessionOptions GetSessionOptions(const OnlineLMConfig &config) {
  return GetSessionOptionsImpl(config.lm_num_threads, config.lm_provider);
}

Ort::SessionOptions GetSessionOptions(const VadModelConfig &config) {
  return GetSessionOptionsImpl(config.num_threads, config.provider);
}

}  // namespace sherpa_onnx
