// sherpa-onnx/csrc/online-transducer-model.cc
//
// Copyright (c)  2023  Xiaomi Corporation
#include "sherpa-onnx/csrc/online-transducer-model.h"

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#include <memory>
#include <sstream>
#include <string>

#include "sherpa-onnx/csrc/online-lstm-transducer-model.h"
#include "sherpa-onnx/csrc/online-zipformer-transducer-model.h"
#include "sherpa-onnx/csrc/onnx-utils.h"
namespace sherpa_onnx {

enum class ModelType {
  kLstm,
  kZipformer,
  kUnkown,
};

static ModelType GetModelType(char *model_data, size_t model_data_length,
                              bool debug) {
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING);
  Ort::SessionOptions sess_opts;

  auto sess = std::make_unique<Ort::Session>(env, model_data, model_data_length,
                                             sess_opts);

  Ort::ModelMetadata meta_data = sess->GetModelMetadata();
  if (debug) {
    std::ostringstream os;
    PrintModelMetadata(os, meta_data);
    fprintf(stderr, "%s\n", os.str().c_str());
  }

  Ort::AllocatorWithDefaultOptions allocator;
  auto model_type =
      meta_data.LookupCustomMetadataMapAllocated("model_type", allocator);
  if (!model_type) {
    fprintf(stderr, "No model_type in the metadata!\n");
    return ModelType::kUnkown;
  }

  if (model_type.get() == std::string("lstm")) {
    return ModelType::kLstm;
  } else if (model_type.get() == std::string("zipformer")) {
    return ModelType::kZipformer;
  } else {
    fprintf(stderr, "Unsupported model_type: %s\n", model_type.get());
    return ModelType::kUnkown;
  }
}

std::unique_ptr<OnlineTransducerModel> OnlineTransducerModel::Create(
    const OnlineTransducerModelConfig &config) {
  auto buffer = ReadFile(config.encoder_filename);

  auto model_type = GetModelType(buffer.data(), buffer.size(), config.debug);

  switch (model_type) {
    case ModelType::kLstm:
      return std::make_unique<OnlineLstmTransducerModel>(config);
    case ModelType::kZipformer:
      return std::make_unique<OnlineZipformerTransducerModel>(config);
    case ModelType::kUnkown:
      return nullptr;
  }

  // unreachable code
  return nullptr;
}

#if __ANDROID_API__ >= 9
std::unique_ptr<OnlineTransducerModel> OnlineTransducerModel::Create(
    AAssetManager *mgr, const OnlineTransducerModelConfig &config) {
  auto buffer = ReadFile(mgr, config.encoder_filename);
  auto model_type = GetModelType(buffer.data(), buffer.size(), config.debug);

  switch (model_type) {
    case ModelType::kLstm:
      return std::make_unique<OnlineLstmTransducerModel>(mgr, config);
    case ModelType::kZipformer:
      return std::make_unique<OnlineZipformerTransducerModel>(mgr, config);
    case ModelType::kUnkown:
      return nullptr;
  }

  // unreachable code
  return nullptr;
}
#endif

}  // namespace sherpa_onnx
