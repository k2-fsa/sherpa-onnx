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
#include "nnapi_provider_factory.h"  // NOLINT
#endif

namespace sherpa_onnx {

static void OrtStatusFailure(OrtStatus *status, const char *s) {
  const auto &api = Ort::GetApi();
  const char *msg = api.GetErrorMessage(status);
  SHERPA_ONNX_LOGE(
      "Failed to enable TensorRT : %s."
      "Available providers: %s. Fallback to cuda",
      msg, s);
  api.ReleaseStatus(status);
}

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
    case Provider::kTRT: {
      struct TrtPairs {
        const char *op_keys;
        const char *op_values;
      };

      std::vector<TrtPairs> trt_options = {
          {"device_id", "0"},
          {"trt_max_workspace_size", "2147483648"},
          {"trt_max_partition_iterations", "10"},
          {"trt_min_subgraph_size", "5"},
          {"trt_fp16_enable", "0"},
          {"trt_detailed_build_log", "0"},
          {"trt_engine_cache_enable", "1"},
          {"trt_engine_cache_path", "."},
          {"trt_timing_cache_enable", "1"},
          {"trt_timing_cache_path", "."}};
      // ToDo : Trt configs
      // "trt_int8_enable"
      // "trt_int8_use_native_calibration_table"
      // "trt_dump_subgraphs"

      std::vector<const char *> option_keys, option_values;
      for (const TrtPairs &pair : trt_options) {
        option_keys.emplace_back(pair.op_keys);
        option_values.emplace_back(pair.op_values);
      }

      std::vector<std::string> available_providers =
          Ort::GetAvailableProviders();
      if (std::find(available_providers.begin(), available_providers.end(),
                    "TensorrtExecutionProvider") != available_providers.end()) {
        const auto &api = Ort::GetApi();

        OrtTensorRTProviderOptionsV2 *tensorrt_options = nullptr;
        OrtStatus *statusC =
            api.CreateTensorRTProviderOptions(&tensorrt_options);
        OrtStatus *statusU = api.UpdateTensorRTProviderOptions(
            tensorrt_options, option_keys.data(), option_values.data(),
            option_keys.size());
        sess_opts.AppendExecutionProvider_TensorRT_V2(*tensorrt_options);

        if (statusC) {
          OrtStatusFailure(statusC, os.str().c_str());
        }
        if (statusU) {
          OrtStatusFailure(statusU, os.str().c_str());
        }

        api.ReleaseTensorRTProviderOptions(tensorrt_options);
      }
      // break; is omitted here intentionally so that
      // if TRT not available, CUDA will be used
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
#elif defined(__ANDROID_API__)
      SHERPA_ONNX_LOGE(
          "Android NNAPI requires API level >= 27. Current API level %d "
          "Fallback to cpu!",
          (int32_t)__ANDROID_API__);
#else
      SHERPA_ONNX_LOGE("NNAPI is for Android only. Fallback to cpu");
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

#if SHERPA_ONNX_ENABLE_TTS
Ort::SessionOptions GetSessionOptions(const OfflineTtsModelConfig &config) {
  return GetSessionOptionsImpl(config.num_threads, config.provider);
}
#endif

Ort::SessionOptions GetSessionOptions(
    const SpeakerEmbeddingExtractorConfig &config) {
  return GetSessionOptionsImpl(config.num_threads, config.provider);
}

Ort::SessionOptions GetSessionOptions(
    const SpokenLanguageIdentificationConfig &config) {
  return GetSessionOptionsImpl(config.num_threads, config.provider);
}

Ort::SessionOptions GetSessionOptions(const AudioTaggingModelConfig &config) {
  return GetSessionOptionsImpl(config.num_threads, config.provider);
}

Ort::SessionOptions GetSessionOptions(
    const OfflinePunctuationModelConfig &config) {
  return GetSessionOptionsImpl(config.num_threads, config.provider);
}

}  // namespace sherpa_onnx
