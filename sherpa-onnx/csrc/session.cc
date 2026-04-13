// sherpa-onnx/csrc/session.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/session.h"

#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/provider.h"
#include "sherpa-onnx/csrc/text-utils.h"
#if defined(__APPLE__) && (ORT_API_VERSION >= 15) && \
    !defined(SHERPA_ONNX_DISABLE_COREML)
#include "coreml_provider_factory.h"  // NOLINT
#endif

#if __ANDROID_API__ >= 27
#include "nnapi_provider_factory.h"  // NOLINT
#endif

#if defined(_WIN32) && SHERPA_ONNX_ENABLE_DIRECTML == 1
#include "dml_provider_factory.h"  // NOLINT
#endif

#if defined(SHERPA_ONNX_ENABLE_SPACEMIT)
#include "spacemit_ort_env.h"  // NOLINT
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

static void ParseConfigFile(
    const std::string &config_path,
    std::unordered_map<std::string, std::string> &config) {
  // The config file should be in the format of key=value per line, e.g.,
  // GraphOptimizationLevel=0 or GraphOptimizationLevel = 0
  // LogSeverityLevel=1
  // ProfilingFilePrefix=profile
  // For spacemit, the config file can contain the following
  // SPACEMIT_EP_USE_GLOBAL_INTRA_THREAD=1
  // ... and so on.
  // # is treated as comment. Empty lines are ignored. The legacy key:value
  // format is still supported for backward compatibility.
  // additionally, DEBUG=1 can be set to print all configs read from the file.

  std::ifstream is(config_path);
  if (!is.is_open()) {
    SHERPA_ONNX_LOGE("Failed to open provider config file: %s",
                     config_path.c_str());
    SHERPA_ONNX_EXIT(-1);
  }

  std::string line;
  int32_t line_num = 0;
  while (std::getline(is, line)) {
    ++line_num;
    line = Trim(line);
    if (line.empty() || line[0] == '#') {
      continue;
    }

    auto sep_pos = line.find('=');
    if (sep_pos == std::string::npos) {
      SHERPA_ONNX_LOGE("Ignore invalid provider config line %d in %s: %s",
                       line_num, config_path.c_str(), line.c_str());
      continue;
    }

    std::string key = Trim(line.substr(0, sep_pos));
    std::string value = Trim(line.substr(sep_pos + 1));

    if (key.empty()) {
      SHERPA_ONNX_LOGE("Ignore provider config line %d with empty key in %s",
                       line_num, config_path.c_str());
      continue;
    }

    config[std::move(key)] = std::move(value);
  }
}

static void SplitProviderAndConfig(
    const std::string &s, std::string &provider_str,
    std::unordered_map<std::string, std::string> &config) {
  // provider string format: provider_name[:config_path], e.g.,
  // tensorrt:trt_config.config or spacemit:spacemit_config.config or cpu. The
  // config file is optional. If it is not provided, default config will be used.
  provider_str = Trim(s);
  config.clear();

  auto pos = provider_str.find(':');
  if (pos == std::string::npos) {
    return;
  }

  std::string config_path = Trim(provider_str.substr(pos + 1));
  provider_str = Trim(provider_str.substr(0, pos));

  if (config_path.empty()) {
    SHERPA_ONNX_LOGE("Provider config path is empty: %s", s.c_str());
    SHERPA_ONNX_EXIT(-1);
  }

  ParseConfigFile(config_path, config);

  if (config.find("DEBUG") != config.end()) {
    auto config_debug = ToIntOrDefault(config["DEBUG"], 0) == 1;
    if (config_debug) {
      SHERPA_ONNX_LOGE("Provider string: %s. Config path: %s",
                       provider_str.c_str(), config_path.c_str());
      for (const auto &kv : config) {
        SHERPA_ONNX_LOGE("Provider config: %s=%s", kv.first.c_str(),
                         kv.second.c_str());
      }
    }
  }
}

Ort::SessionOptions GetSessionOptionsImpl(
    int32_t num_threads, const std::string &provider_str,
    const ProviderConfig *provider_config /*= nullptr*/) {
  std::unordered_map<std::string, std::string> config;
  std::string new_provider_str;

  SplitProviderAndConfig(provider_str, new_provider_str, config);

  Provider p = StringToProvider(new_provider_str);

  Ort::SessionOptions sess_opts;
  sess_opts.SetIntraOpNumThreads(num_threads);

  sess_opts.SetInterOpNumThreads(num_threads);

  std::vector<std::string> available_providers = Ort::GetAvailableProviders();
  std::ostringstream os;
  for (const auto &ep : available_providers) {
    os << ep << ", ";
  }

  // Other possible options
  if (config.find("GraphOptimizationLevel") != config.end()) {
    int32_t graph_optimization_level = ToIntOrDefault(
        config["GraphOptimizationLevel"],
        static_cast<int32_t>(GraphOptimizationLevel::ORT_ENABLE_ALL));
    sess_opts.SetGraphOptimizationLevel(
        static_cast<GraphOptimizationLevel>(graph_optimization_level));
    config.erase("GraphOptimizationLevel");
  }

  if (config.find("LogSeverityLevel") != config.end()) {
    int32_t log_severity_level = ToIntOrDefault(
        config["LogSeverityLevel"],
        static_cast<int32_t>(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING));
    sess_opts.SetLogSeverityLevel(
        static_cast<OrtLoggingLevel>(log_severity_level));
    config.erase("LogSeverityLevel");
  }

  if (config.find("ProfilingFilePrefix") != config.end()) {
    sess_opts.EnableProfiling(SHERPA_ONNX_TO_ORT_PATH(config["ProfilingFilePrefix"]));
    config.erase("ProfilingFilePrefix");
  }

  if (config.find("EnableMemPattern") != config.end()) {
    int32_t enable_mem_pattern =
        ToIntOrDefault(config["EnableMemPattern"], 1);
    if (enable_mem_pattern == 0) {
      sess_opts.DisableMemPattern();
    }
    config.erase("EnableMemPattern");
  }

  if (config.find("EnableCpuMemArena") != config.end()) {
    int32_t enable_cpu_mem_arena =
        ToIntOrDefault(config["EnableCpuMemArena"], 1);
    if (enable_cpu_mem_arena == 0) {
      sess_opts.DisableCpuMemArena();
    }
    config.erase("EnableCpuMemArena");
  }

  // If you want to speed up initialization, please uncomment the following line
  // sess_opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);

  switch (p) {
    case Provider::kCPU:
      break;  // nothing to do for the CPU provider
    case Provider::kXnnpack: {
#if ORT_API_VERSION >= 12
      if (std::find(available_providers.begin(), available_providers.end(),
                    "XnnpackExecutionProvider") != available_providers.end()) {
        sess_opts.AppendExecutionProvider("XNNPACK");
      } else {
        SHERPA_ONNX_LOGE("Available providers: %s. Fallback to cpu!",
                         os.str().c_str());
      }
#else
      SHERPA_ONNX_LOGE(
          "Does not support xnnpack for onnxruntime: %d. Fallback to cpu!",
          static_cast<int32_t>(ORT_API_VERSION));
#endif
      break;
    }
    case Provider::kTRT: {
      if (provider_config == nullptr) {
        SHERPA_ONNX_LOGE(
            "Tensorrt support for Online models only,"
            "Must be extended for offline and others");
        SHERPA_ONNX_EXIT(1);
      }
      auto trt_config = provider_config->trt_config;
      struct TrtPairs {
        const char *op_keys;
        const char *op_values;
      };

      auto device_id = std::to_string(provider_config->device);
      auto trt_max_workspace_size =
          std::to_string(trt_config.trt_max_workspace_size);
      auto trt_max_partition_iterations =
          std::to_string(trt_config.trt_max_partition_iterations);
      auto trt_min_subgraph_size =
          std::to_string(trt_config.trt_min_subgraph_size);
      auto trt_fp16_enable = std::to_string(trt_config.trt_fp16_enable);
      auto trt_detailed_build_log =
          std::to_string(trt_config.trt_detailed_build_log);
      auto trt_engine_cache_enable =
          std::to_string(trt_config.trt_engine_cache_enable);
      auto trt_timing_cache_enable =
          std::to_string(trt_config.trt_timing_cache_enable);
      auto trt_dump_subgraphs = std::to_string(trt_config.trt_dump_subgraphs);
      std::vector<TrtPairs> trt_options = {
          {"device_id", device_id.c_str()},
          {"trt_max_workspace_size", trt_max_workspace_size.c_str()},
          {"trt_max_partition_iterations",
           trt_max_partition_iterations.c_str()},
          {"trt_min_subgraph_size", trt_min_subgraph_size.c_str()},
          {"trt_fp16_enable", trt_fp16_enable.c_str()},
          {"trt_detailed_build_log", trt_detailed_build_log.c_str()},
          {"trt_engine_cache_enable", trt_engine_cache_enable.c_str()},
          {"trt_engine_cache_path", trt_config.trt_engine_cache_path.c_str()},
          {"trt_timing_cache_enable", trt_timing_cache_enable.c_str()},
          {"trt_timing_cache_path", trt_config.trt_timing_cache_path.c_str()},
          {"trt_dump_subgraphs", trt_dump_subgraphs.c_str()}};
      // ToDo : Trt configs
      // "trt_int8_enable"
      // "trt_int8_use_native_calibration_table"

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

        if (provider_config != nullptr) {
          options.device_id = provider_config->device;
          options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearch(
              provider_config->cuda_config.cudnn_conv_algo_search);
        } else {
          options.device_id = 0;
          // Default OrtCudnnConvAlgoSearchExhaustive is extremely slow
          options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchHeuristic;
          // set more options on need
        }
        sess_opts.AppendExecutionProvider_CUDA(options);
      } else {
        SHERPA_ONNX_LOGE(
            "Please compile with -DSHERPA_ONNX_ENABLE_GPU=ON. Available "
            "providers: %s. Fallback to cpu!",
            os.str().c_str());
      }
      break;
    }
    case Provider::kDirectML: {
#if defined(_WIN32) && SHERPA_ONNX_ENABLE_DIRECTML == 1
      sess_opts.DisableMemPattern();
      sess_opts.SetExecutionMode(ORT_SEQUENTIAL);
      int32_t device_id = 0;
      OrtStatus *status =
          OrtSessionOptionsAppendExecutionProvider_DML(sess_opts, device_id);
      if (status) {
        const auto &api = Ort::GetApi();
        const char *msg = api.GetErrorMessage(status);
        SHERPA_ONNX_LOGE("Failed to enable DirectML: %s. Fallback to cpu", msg);
        api.ReleaseStatus(status);
      }
#else
      SHERPA_ONNX_LOGE("DirectML is for Windows only. Fallback to cpu!");
#endif
      break;
    }
    case Provider::kCoreML: {
#if defined(__APPLE__) && (ORT_API_VERSION >= 15) && \
    !defined(SHERPA_ONNX_DISABLE_COREML)
      uint32_t coreml_flags = 0;
      (void)OrtSessionOptionsAppendExecutionProvider_CoreML(sess_opts,
                                                            coreml_flags);
#else
      SHERPA_ONNX_LOGE(
          "CoreML is for Apple only since onnxruntime>=1.15. Fallback to cpu!");
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
    case Provider::kSpacemiT: {
#if defined(SHERPA_ONNX_ENABLE_SPACEMIT)
      SHERPA_ONNX_LOGE("Use SpacemiT Execution Provider");
      // when using SpacemiT Execution Provider, set intra_op_num_threads and
      // inter_op_num_threads to 1 can improve performance.
      // all ops run on ep, no need to create multiple threads in onnxruntime.
      // ep will create SPACEMIT_EP_INTRA_THREAD_NUM threads as intra threads.
      std::unordered_map<std::string, std::string> provider_options(config);
      SHERPA_ONNX_LOGE("Set IntraOpNumThreads to 1");
      sess_opts.SetIntraOpNumThreads(1);
      SHERPA_ONNX_LOGE("Set InterOpNumThreads to 1");
      sess_opts.SetInterOpNumThreads(1);

      if (provider_options.find("SPACEMIT_EP_INTRA_THREAD_NUM") ==
          provider_options.end()) {
        SHERPA_ONNX_LOGE("Set SPACEMIT_EP_INTRA_THREAD_NUM to %d", num_threads);
        provider_options.emplace("SPACEMIT_EP_INTRA_THREAD_NUM",
                                 std::to_string(num_threads));
      }
      OrtStatus *sts =
          Ort::SessionOptionsSpaceMITEnvInit(sess_opts, provider_options);
      if (sts) {
        const auto &api = Ort::GetApi();
        const char *msg = api.GetErrorMessage(sts);
        SHERPA_ONNX_LOGE(
            "Failed to enable SpacemiT Execution Provider: %s. Fallback to cpu",
            msg);
        api.ReleaseStatus(sts);
      }
#else
      SHERPA_ONNX_LOGE(
          "SpacemiT Execution Provider is for SpacemiT AI-CPUs only. Fallback "
          "to cpu!");
#endif
      break;
    }
  }
  return sess_opts;
}

Ort::SessionOptions GetSessionOptions(const OnlineModelConfig &config) {
  return GetSessionOptionsImpl(config.num_threads,
                               config.provider_config.provider,
                               &config.provider_config);
}

Ort::SessionOptions GetSessionOptions(const OnlineModelConfig &config,
                                      const std::string &model_type) {
  /*
    Transducer models : Only encoder will run with tensorrt,
                        decoder and joiner will run with cuda
  */
  if (config.provider_config.provider == "trt" &&
      (model_type == "decoder" || model_type == "joiner")) {
    return GetSessionOptionsImpl(config.num_threads, "cuda",
                                 &config.provider_config);
  }
  return GetSessionOptionsImpl(config.num_threads,
                               config.provider_config.provider,
                               &config.provider_config);
}

Ort::SessionOptions GetSessionOptions(const OfflineLMConfig &config) {
  return GetSessionOptionsImpl(config.lm_num_threads, config.lm_provider);
}

Ort::SessionOptions GetSessionOptions(const OnlineLMConfig &config) {
  return GetSessionOptionsImpl(config.lm_num_threads, config.lm_provider);
}

Ort::SessionOptions GetSessionOptions(int32_t num_threads,
                                      const std::string &provider_str) {
  return GetSessionOptionsImpl(num_threads, provider_str);
}

}  // namespace sherpa_onnx
