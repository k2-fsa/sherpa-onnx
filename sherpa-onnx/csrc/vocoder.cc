// sherpa-onnx/csrc/vocoder.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa-onnx/csrc/vocoder.h"

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#if __OHOS__
#include "rawfile/raw_file_manager.h"
#endif

#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/hifigan-vocoder.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/onnx-utils.h"
#include "sherpa-onnx/csrc/vocos-vocoder.h"

namespace sherpa_onnx {

namespace {

enum class ModelType : std::uint8_t {
  kHifigan,
  kVocoos,
  kUnknown,
};

}  // namespace

static ModelType GetModelType(char *model_data, size_t model_data_length,
                              bool debug) {
  Ort::Env env(ORT_LOGGING_LEVEL_ERROR);
  Ort::SessionOptions sess_opts;
  sess_opts.SetIntraOpNumThreads(1);
  sess_opts.SetInterOpNumThreads(1);

  auto sess = std::make_unique<Ort::Session>(env, model_data, model_data_length,
                                             sess_opts);

  Ort::ModelMetadata meta_data = sess->GetModelMetadata();
  if (debug) {
    std::ostringstream os;
    PrintModelMetadata(os, meta_data);
#if __OHOS__
    SHERPA_ONNX_LOGE("%{public}s", os.str().c_str());
#else
    SHERPA_ONNX_LOGE("%s", os.str().c_str());
#endif
  }

  Ort::AllocatorWithDefaultOptions allocator;
  auto model_type =
      LookupCustomModelMetaData(meta_data, "model_type", allocator);
  if (model_type.empty()) {
    SHERPA_ONNX_LOGE(
        "No model_type in the metadata!\n"
        "Please make sure you are using the vocoder from "
        "https://github.com/k2-fsa/sherpa-onnx/releases/tag/vocoder-models");
    return ModelType::kUnknown;
  }

  if (model_type == "hifigan") {
    return ModelType::kHifigan;
  } else if (model_type == "vocos") {
    return ModelType::kVocoos;
  } else {
    SHERPA_ONNX_LOGE("Unsupported model_type: %s", model_type.c_str());
    return ModelType::kUnknown;
  }
}

std::unique_ptr<Vocoder> Vocoder::Create(const OfflineTtsModelConfig &config) {
  auto buffer = ReadFile(config.matcha.vocoder);
  auto model_type = GetModelType(buffer.data(), buffer.size(), config.debug);

  switch (model_type) {
    case ModelType::kHifigan:
      return std::make_unique<HifiganVocoder>(
          config.num_threads, config.provider, config.matcha.vocoder);
    case ModelType::kVocoos:
      return std::make_unique<VocosVocoder>(config);
    case ModelType::kUnknown:
      SHERPA_ONNX_LOGE("Unknown model type in vocoder!");
      return nullptr;
  }

  return nullptr;
}

template <typename Manager>
std::unique_ptr<Vocoder> Vocoder::Create(Manager *mgr,
                                         const OfflineTtsModelConfig &config) {
  auto buffer = ReadFile(mgr, config.matcha.vocoder);
  auto model_type = GetModelType(buffer.data(), buffer.size(), config.debug);

  switch (model_type) {
    case ModelType::kHifigan:
      return std::make_unique<HifiganVocoder>(
          mgr, config.num_threads, config.provider, config.matcha.vocoder);
    case ModelType::kVocoos:
      return std::make_unique<VocosVocoder>(mgr, config);
    case ModelType::kUnknown:
      SHERPA_ONNX_LOGE("Unknown model type in vocoder!");
      return nullptr;
  }
}

#if __ANDROID_API__ >= 9
template std::unique_ptr<Vocoder> Vocoder::Create(
    AAssetManager *mgr, const OfflineTtsModelConfig &config);
#endif

#if __OHOS__
template std::unique_ptr<Vocoder> Vocoder::Create(
    NativeResourceManager *mgr, const OfflineTtsModelConfig &config);
#endif

}  // namespace sherpa_onnx
