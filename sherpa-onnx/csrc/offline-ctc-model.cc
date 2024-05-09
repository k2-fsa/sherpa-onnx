// sherpa-onnx/csrc/offline-ctc-model.cc
//
// Copyright (c)  2022-2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-ctc-model.h"

#include <algorithm>
#include <memory>
#include <sstream>
#include <string>

#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/offline-nemo-enc-dec-ctc-model.h"
#include "sherpa-onnx/csrc/offline-tdnn-ctc-model.h"
#include "sherpa-onnx/csrc/offline-wenet-ctc-model.h"
#include "sherpa-onnx/csrc/offline-zipformer-ctc-model.h"
#include "sherpa-onnx/csrc/onnx-utils.h"

namespace {

enum class ModelType {
  kEncDecCTCModelBPE,
  kEncDecHybridRNNTCTCBPEModel,
  kTdnn,
  kZipformerCtc,
  kWenetCtc,
  kUnknown,
};

}  // namespace

namespace sherpa_onnx {

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
      meta_data.LookupCustomMetadataMapAllocated("model_type", allocator);
  if (!model_type) {
    SHERPA_ONNX_LOGE(
        "No model_type in the metadata!\n"
        "If you are using models from NeMo, please refer to\n"
        "https://huggingface.co/csukuangfj/"
        "sherpa-onnx-nemo-ctc-en-citrinet-512/blob/main/add-model-metadata.py\n"
        "or "
        "https://github.com/k2-fsa/sherpa-onnx/tree/master/scripts/nemo/"
        "fast-conformer-hybrid-transducer-ctc\n"
        "If you are using models from WeNet, please refer to\n"
        "https://github.com/k2-fsa/sherpa-onnx/blob/master/scripts/wenet/"
        "run.sh\n"
        "\n"
        "for how to add metadta to model.onnx\n");
    return ModelType::kUnknown;
  }

  if (model_type.get() == std::string("EncDecCTCModelBPE")) {
    return ModelType::kEncDecCTCModelBPE;
  } else if (model_type.get() == std::string("EncDecHybridRNNTCTCBPEModel")) {
    return ModelType::kEncDecHybridRNNTCTCBPEModel;
  } else if (model_type.get() == std::string("tdnn")) {
    return ModelType::kTdnn;
  } else if (model_type.get() == std::string("zipformer2_ctc")) {
    return ModelType::kZipformerCtc;
  } else if (model_type.get() == std::string("wenet_ctc")) {
    return ModelType::kWenetCtc;
  } else {
    SHERPA_ONNX_LOGE("Unsupported model_type: %s", model_type.get());
    return ModelType::kUnknown;
  }
}

std::unique_ptr<OfflineCtcModel> OfflineCtcModel::Create(
    const OfflineModelConfig &config) {
  ModelType model_type = ModelType::kUnknown;

  std::string filename;
  if (!config.nemo_ctc.model.empty()) {
    filename = config.nemo_ctc.model;
  } else if (!config.tdnn.model.empty()) {
    filename = config.tdnn.model;
  } else if (!config.zipformer_ctc.model.empty()) {
    filename = config.zipformer_ctc.model;
  } else if (!config.wenet_ctc.model.empty()) {
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
    case ModelType::kEncDecCTCModelBPE:
      return std::make_unique<OfflineNemoEncDecCtcModel>(config);
      break;
    case ModelType::kEncDecHybridRNNTCTCBPEModel:
      return std::make_unique<OfflineNemoEncDecHybridRNNTCTCBPEModel>(config);
      break;
    case ModelType::kTdnn:
      return std::make_unique<OfflineTdnnCtcModel>(config);
      break;
    case ModelType::kZipformerCtc:
      return std::make_unique<OfflineZipformerCtcModel>(config);
      break;
    case ModelType::kWenetCtc:
      return std::make_unique<OfflineWenetCtcModel>(config);
      break;
    case ModelType::kUnknown:
      SHERPA_ONNX_LOGE("Unknown model type in offline CTC!");
      return nullptr;
  }

  return nullptr;
}

#if __ANDROID_API__ >= 9

std::unique_ptr<OfflineCtcModel> OfflineCtcModel::Create(
    AAssetManager *mgr, const OfflineModelConfig &config) {
  ModelType model_type = ModelType::kUnknown;

  std::string filename;
  if (!config.nemo_ctc.model.empty()) {
    filename = config.nemo_ctc.model;
  } else if (!config.tdnn.model.empty()) {
    filename = config.tdnn.model;
  } else if (!config.zipformer_ctc.model.empty()) {
    filename = config.zipformer_ctc.model;
  } else if (!config.wenet_ctc.model.empty()) {
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
    case ModelType::kEncDecCTCModelBPE:
      return std::make_unique<OfflineNemoEncDecCtcModel>(mgr, config);
      break;
    case ModelType::kEncDecHybridRNNTCTCBPEModel:
      return std::make_unique<OfflineNemoEncDecHybridRNNTCTCBPEModel>(config);
      break;
    case ModelType::kTdnn:
      return std::make_unique<OfflineTdnnCtcModel>(mgr, config);
      break;
    case ModelType::kZipformerCtc:
      return std::make_unique<OfflineZipformerCtcModel>(mgr, config);
      break;
    case ModelType::kWenetCtc:
      return std::make_unique<OfflineWenetCtcModel>(mgr, config);
      break;
    case ModelType::kUnknown:
      SHERPA_ONNX_LOGE("Unknown model type in offline CTC!");
      return nullptr;
  }

  return nullptr;
}
#endif

}  // namespace sherpa_onnx
