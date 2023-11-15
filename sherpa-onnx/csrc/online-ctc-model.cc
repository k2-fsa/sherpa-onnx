// sherpa-onnx/csrc/online-ctc-model.cc
//
// Copyright (c)  2022-2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/online-ctc-model.h"

#include <algorithm>
#include <memory>
#include <sstream>
#include <string>

#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/online-wenet-ctc-model.h"
#include "sherpa-onnx/csrc/onnx-utils.h"

namespace {

enum class ModelType {
  kZipformerCtc,
  kWenetCtc,
  kUnkown,
};

}  // namespace

namespace sherpa_onnx {

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
    SHERPA_ONNX_LOGE("%s", os.str().c_str());
  }

  Ort::AllocatorWithDefaultOptions allocator;
  auto model_type =
      meta_data.LookupCustomMetadataMapAllocated("model_type", allocator);
  if (!model_type) {
    SHERPA_ONNX_LOGE(
        "No model_type in the metadata!\n"
        "If you are using models from WeNet, please refer to\n"
        "https://github.com/k2-fsa/sherpa-onnx/blob/master/scripts/wenet/"
        "run.sh\n"
        "\n"
        "for how to add metadta to model.onnx\n");
    return ModelType::kUnkown;
  }

  if (model_type.get() == std::string("zipformer2")) {
    return ModelType::kZipformerCtc;
  } else if (model_type.get() == std::string("wenet_ctc")) {
    return ModelType::kWenetCtc;
  } else {
    SHERPA_ONNX_LOGE("Unsupported model_type: %s", model_type.get());
    return ModelType::kUnkown;
  }
}

std::unique_ptr<OnlineCtcModel> OnlineCtcModel::Create(
    const OnlineModelConfig &config) {
  ModelType model_type = ModelType::kUnkown;

  std::string filename;
  if (!config.wenet_ctc.model.empty()) {
    filename = config.wenet_ctc.model;
  } else {
    SHERPA_ONNX_LOGE("Please specify a CTC model");
    exit(-1);
  }

  {
    auto buffer = ReadFile(filename);

    model_type = GetModelType(buffer.data(), buffer.size(), config.debug);
  }

  switch (model_type) {
    case ModelType::kZipformerCtc:
      return nullptr;
      // return std::make_unique<OnlineZipformerCtcModel>(config);
      break;
    case ModelType::kWenetCtc:
      return std::make_unique<OnlineWenetCtcModel>(config);
      break;
    case ModelType::kUnkown:
      SHERPA_ONNX_LOGE("Unknown model type in online CTC!");
      return nullptr;
  }

  return nullptr;
}

#if __ANDROID_API__ >= 9

std::unique_ptr<OnlineCtcModel> OnlineCtcModel::Create(
    AAssetManager *mgr, const OnlineModelConfig &config) {
  ModelType model_type = ModelType::kUnkown;

  std::string filename;
  if (!config.wenet_ctc.model.empty()) {
    filename = config.wenet_ctc.model;
  } else {
    SHERPA_ONNX_LOGE("Please specify a CTC model");
    exit(-1);
  }

  {
    auto buffer = ReadFile(mgr, filename);

    model_type = GetModelType(buffer.data(), buffer.size(), config.debug);
  }

  switch (model_type) {
    case ModelType::kZipformerCtc:
      return nullptr;
      // return std::make_unique<OnlineZipformerCtcModel>(mgr, config);
      break;
    case ModelType::kWenetCtc:
      return std::make_unique<OnlineWenetCtcModel>(mgr, config);
      break;
    case ModelType::kUnkown:
      SHERPA_ONNX_LOGE("Unknown model type in online CTC!");
      return nullptr;
  }

  return nullptr;
}
#endif

}  // namespace sherpa_onnx
