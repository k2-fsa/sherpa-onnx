// sherpa-onnx/csrc/offline-recognizer-impl.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-recognizer-impl.h"

#include <string>

#include "onnxruntime_cxx_api.h"  // NOLINT
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/offline-recognizer-transducer-impl.h"
#include "sherpa-onnx/csrc/onnx-utils.h"
#include "sherpa-onnx/csrc/text-utils.h"

namespace sherpa_onnx {

std::unique_ptr<OfflineRecognizerImpl> OfflineRecognizerImpl::Create(
    const OfflineRecognizerConfig &config) {
  Ort::Env env;

  Ort::SessionOptions sess_opts;
  auto buf = ReadFile(config.model_config.encoder_filename);

  auto encoder_sess =
      std::make_unique<Ort::Session>(env, buf.data(), buf.size(), sess_opts);

  Ort::ModelMetadata meta_data = encoder_sess->GetModelMetadata();

  Ort::AllocatorWithDefaultOptions allocator;  // used in the macro below

  std::string model_type;
  SHERPA_ONNX_READ_META_DATA_STR(model_type, "model_type");

  if (model_type == "conformer") {
    return std::make_unique<OfflineRecognizerTransducerImpl>(config);
  }

  SHERPA_ONNX_LOGE("Unsupported model_type: %s\n", model_type.c_str());

  exit(-1);
}

}  // namespace sherpa_onnx
