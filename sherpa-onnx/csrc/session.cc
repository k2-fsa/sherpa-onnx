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

namespace {

struct SessionConfig {
  int32_t num_threads;
  std::string provider;
  uint32_t nnapi_flags = 0;
};

}  // namespace

static Ort::SessionOptions GetSessionOptionsImpl(SessionConfig config) {
  Provider p = StringToProvider(std::move(config.provider));

  Ort::SessionOptions sess_opts;
  sess_opts.SetIntraOpNumThreads(config.num_threads);
  sess_opts.SetInterOpNumThreads(config.num_threads);

  const auto &api = Ort::GetApi();

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
      OrtStatus *status = OrtSessionOptionsAppendExecutionProvider_CoreML(
          sess_opts, coreml_flags);
      if (status) {
        const char *msg = api.GetErrorMessage(status);
        SHERPA_ONNX_LOGE("Failed to enable CoreML: %s. Fallback to cpu", msg);
        api.ReleaseStatus(status);
      }
#else
      SHERPA_ONNX_LOGE("CoreML is for Apple only. Fallback to cpu!");
#endif
      break;
    }
    case Provider::kNNAPI: {
#if __ANDROID_API__ >= 27
      SHERPA_ONNX_LOGE("Current API level %d ", (int32_t)__ANDROID_API__);
      OrtStatus *status = OrtSessionOptionsAppendExecutionProvider_Nnapi(
          sess_opts, config.nnapi_flags);
      if (status) {
        const char *msg = api.GetErrorMessage(status);  // causes segfault
        SHERPA_ONNX_LOGE("Failed to enable NNAPI: %s. Fallback to cpu", msg);
        api.ReleaseStatus(status);
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

Ort::SessionOptions GetSessionOptions(
    const OnlineTransducerModelConfig &config) {
  SessionConfig sess_config;

  sess_config.num_threads = config.num_threads;
  sess_config.provider = config.provider;
  sess_config.nnapi_flags = config.nnapi_flags;

  return GetSessionOptionsImpl(std::move(sess_config));
}

Ort::SessionOptions GetSessionOptions(const OfflineModelConfig &config) {
  SessionConfig sess_config;

  sess_config.num_threads = config.num_threads;
  sess_config.provider = config.provider;
  sess_config.nnapi_flags = config.nnapi_flags;

  return GetSessionOptionsImpl(std::move(sess_config));
}

}  // namespace sherpa_onnx
