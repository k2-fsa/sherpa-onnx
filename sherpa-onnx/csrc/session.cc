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

#if __ANDROID_API__ >= 27
#include "nnapi_provider_factory.h"
#endif

namespace sherpa_onnx {

static Ort::SessionOptions GetSessionOptionsImpl(int32_t num_threads,
                                                 std::string provider_str) {
  Provider p = StringToProvider(std::move(provider_str));

  Ort::SessionOptions sess_opts;
  sess_opts.SetIntraOpNumThreads(num_threads);
  sess_opts.SetInterOpNumThreads(num_threads);

  std::vector<std::string> available_providers = Ort::GetAvailableProviders();
  std::ostringstream os;
  for (const auto &ep : available_providers) {
    os << ep << ", ";
  }

  // Other possible options
  // sess_opts.SetGraphOptimizationLevel(ORT_ENABLE_EXTENDED);
  // sess_opts.SetLogSeverityLevel(ORT_LOGGING_LEVEL_VERBOSE);
  // sess_opts.EnableProfiling("profile");

  switch (p) {
    case Provider::kCPU:
      break;  // nothing to do for the CPU provider
    case Provider::kXnnpack: {
      if (std::find(available_providers.begin(), available_providers.end(),
                    "XnnpackExecutionProvider") != available_providers.end()) {
        sess_opts.AppendExecutionProvider("XNNPACK");
      } else {
        SHERPA_ONNX_LOGE("Available providers: %s. Fallback to cpu!",
                         os.str().c_str());
      }
      break;
    }
    case Provider::kCUDA: {
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
            "Please compile with -DSHERPA_ONNX_ENABLE_GPU=ON. Available "
            "providers: %s. Fallback to cpu!",
            os.str().c_str());
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
    case Provider::kNNAPI: {
#if __ANDROID_API__ >= 27
      SHERPA_ONNX_LOGE("Current API level %d ", (int32_t)__ANDROID_API__);

      // Please see
      // https://onnxruntime.ai/docs/execution-providers/NNAPI-ExecutionProvider.html#usage
      // to enable different flags
      uint32_t nnapi_flags = 0;
      // nnapi_flags |= NNAPI_FLAG_USE_FP16;
      // nnapi_flags |= NNAPI_FLAG_CPU_DISABLED;
      OrtStatus *status = OrtSessionOptionsAppendExecutionProvider_Nnapi(
          sess_opts, nnapi_flags);

      if (status) {
        const auto &api = Ort::GetApi();
        const char *msg = api.GetErrorMessage(status);
        SHERPA_ONNX_LOGE(
            "Failed to enable NNAPI: %s. Available providers: %s. Fallback to "
            "cpu",
            msg, os.str().c_str());
        api.ReleaseStatus(status);
      } else {
        SHERPA_ONNX_LOGE("Use nnapi");
      }
#else
      SHERPA_ONNX_LOGE(
          "Android NNAPI requires API level >= 27. Current API level %d "
          "Fallback to cpu!",
          (int32_t)__ANDROID_API__);
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

Ort::SessionOptions GetSessionOptions(const OfflineTtsModelConfig &config) {
  return GetSessionOptionsImpl(config.num_threads, config.provider);
}

Ort::SessionOptions GetSessionOptions(
    const SpeakerEmbeddingExtractorConfig &config) {
  return GetSessionOptionsImpl(config.num_threads, config.provider);
}

}  // namespace sherpa_onnx
