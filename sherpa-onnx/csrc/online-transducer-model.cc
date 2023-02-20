// sherpa-onnx/csrc/online-transducer-model.cc
//
// Copyright (c)  2023  Xiaomi Corporation
#include "sherpa-onnx/csrc/online-transducer-model.h"

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

static ModelType GetModelType(const OnlineTransducerModelConfig &config) {
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING);
  Ort::SessionOptions sess_opts;

  auto sess = std::make_unique<Ort::Session>(
      env, SHERPA_MAYBE_WIDE(config.encoder_filename).c_str(), sess_opts);

  Ort::ModelMetadata meta_data = sess->GetModelMetadata();
  if (config.debug) {
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
  auto model_type = GetModelType(config);

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

}  // namespace sherpa_onnx
