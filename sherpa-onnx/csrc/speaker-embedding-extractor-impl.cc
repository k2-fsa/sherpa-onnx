// sherpa-onnx/csrc/speaker-embedding-extractor-impl.cc
//
// Copyright (c)  2024  Xiaomi Corporation
#include "sherpa-onnx/csrc/speaker-embedding-extractor-impl.h"

#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/onnx-utils.h"
#include "sherpa-onnx/csrc/speaker-embedding-extractor-general-impl.h"
#include "sherpa-onnx/csrc/speaker-embedding-extractor-nemo-impl.h"

namespace sherpa_onnx {

namespace {

enum class ModelType {
  kWeSpeaker,
  k3dSpeaker,
  kNeMo,
  kUnknown,
};

}  // namespace

static ModelType GetModelType(char *model_data, size_t model_data_length,
                              bool debug) {
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING);
  Ort::SessionOptions sess_opts;
  sess_opts.SetIntraOpNumThreads(1);
  sess_opts.SetInterOpNumThreads(1);

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
      meta_data.LookupCustomMetadataMapAllocated("framework", allocator);
  if (!model_type) {
    SHERPA_ONNX_LOGE(
        "No model_type in the metadata!\n"
        "Please make sure you have added metadata to the model.\n\n"
        "For instance, you can use\n"
        "https://github.com/k2-fsa/sherpa-onnx/blob/master/scripts/wespeaker/"
        "add_meta_data.py"
        "to add metadata to models from WeSpeaker\n");
    return ModelType::kUnknown;
  }

  if (model_type.get() == std::string("wespeaker")) {
    return ModelType::kWeSpeaker;
  } else if (model_type.get() == std::string("3d-speaker")) {
    return ModelType::k3dSpeaker;
  } else if (model_type.get() == std::string("nemo")) {
    return ModelType::kNeMo;
  } else {
    SHERPA_ONNX_LOGE("Unsupported model_type: %s", model_type.get());
    return ModelType::kUnknown;
  }
}

std::unique_ptr<SpeakerEmbeddingExtractorImpl>
SpeakerEmbeddingExtractorImpl::Create(
    const SpeakerEmbeddingExtractorConfig &config) {
  ModelType model_type = ModelType::kUnknown;

  {
    auto buffer = ReadFile(config.model);

    model_type = GetModelType(buffer.data(), buffer.size(), config.debug);
  }

  switch (model_type) {
    case ModelType::kWeSpeaker:
      // fall through
    case ModelType::k3dSpeaker:
      return std::make_unique<SpeakerEmbeddingExtractorGeneralImpl>(config);
    case ModelType::kNeMo:
      return std::make_unique<SpeakerEmbeddingExtractorNeMoImpl>(config);
    case ModelType::kUnknown:
      SHERPA_ONNX_LOGE("Unknown model type for speaker embedding extractor!");
      return nullptr;
  }

  // unreachable code
  return nullptr;
}

#if __ANDROID_API__ >= 9
std::unique_ptr<SpeakerEmbeddingExtractorImpl>
SpeakerEmbeddingExtractorImpl::Create(
    AAssetManager *mgr, const SpeakerEmbeddingExtractorConfig &config) {
  ModelType model_type = ModelType::kUnknown;

  {
    auto buffer = ReadFile(mgr, config.model);

    model_type = GetModelType(buffer.data(), buffer.size(), config.debug);
  }

  switch (model_type) {
    case ModelType::kWeSpeaker:
      // fall through
    case ModelType::k3dSpeaker:
      return std::make_unique<SpeakerEmbeddingExtractorGeneralImpl>(mgr,
                                                                    config);
    case ModelType::kNeMo:
      return std::make_unique<SpeakerEmbeddingExtractorNeMoImpl>(mgr, config);
    case ModelType::kUnknown:
      SHERPA_ONNX_LOGE(
          "Unknown model type in for speaker embedding extractor!");
      return nullptr;
  }

  // unreachable code
  return nullptr;
}
#endif

}  // namespace sherpa_onnx
