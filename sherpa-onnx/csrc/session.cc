// sherpa-onnx/csrc/session.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/session.h"

#include <string.h>

#include <string>
#include <utility>

#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/provider.h"
#if defined(__APPLE__)
#include "coreml_provider_factory.h"  // NOLINT
#endif

#if defined(_WIN32)
#include "dml_provider_factory.h"  // NOLINT
#endif

namespace sherpa_onnx {

static Ort::SessionOptions GetSessionOptionsImpl(int32_t num_threads,
                                                 std::string provider_str) {
  Provider p = StringToProvider(std::move(provider_str));

  Ort::SessionOptions sess_opts;
  sess_opts.SetIntraOpNumThreads(num_threads);
  sess_opts.SetInterOpNumThreads(num_threads);

  const auto &api = Ort::GetApi();

  switch (p) {
    case Provider::kCPU:
      break;  // nothing to do for the CPU provider
    case Provider::kCUDA: {
      OrtCUDAProviderOptions options;
      options.device_id = 0;

      // set more options on need
      sess_opts.AppendExecutionProvider_CUDA(options);
      break;
    }
    case Provider::kCoreML: {
#if defined(__APPLE__)
      uint32_t coreml_flags = 0;
      OrtStatus *status = OrtSessionOptionsAppendExecutionProvider_CoreML(
          sess_opts, coreml_flags);
      if (!status) {
        const char *msg = api.GetErrorMessage(status);
        SHERPA_ONNX_LOGE("Failed to enable CoreML: %s. Fallback to cpu", msg);
        api.ReleaseStatus(status);
      }
#else
      SHERPA_ONNX_LOGE("CoreML is for Apple only. Fallback to cpu!");
#endif
      break;
    }
    case Provider::kDirectML: {
#if defined(_WIN32)
      sess_opts.DisableMemPattern();
      sess_opts.SetExecutionMode(ORT_SEQUENTIAL);
      int32_t device_id = 0;
      OrtStatus *status =
          OrtSessionOptionsAppendExecutionProvider_DML(sess_opts, device_id);
      if (!status) {
        const char *msg = api.GetErrorMessage(status);
        SHERPA_ONNX_LOGE("Failed to enable DirectML: %s. Fallback to cpu", msg);
        api.ReleaseStatus(status);
      }
#else
      SHERPA_ONNX_LOGE("DirectML is for Windows only. Fallback to cpu!");
#endif
      break;
    }
  }

  return sess_opts;
}

Ort::SessionOptions GetSessionOptions(
    const OnlineTransducerModelConfig &config) {
  return GetSessionOptionsImpl(config.num_threads, config.provider);
}

Ort::SessionOptions GetSessionOptions(const OfflineModelConfig &config) {
  return GetSessionOptionsImpl(config.num_threads, config.provider);
}

}  // namespace sherpa_onnx
