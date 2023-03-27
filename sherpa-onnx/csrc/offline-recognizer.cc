// sherpa-onnx/csrc/offline-recognizer.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-recognizer.h"

#include <memory>

#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/offline-recognizer-impl.h"

namespace sherpa_onnx {

void OfflineRecognizerConfig::Register(ParseOptions *po) {
  feat_config.Register(po);
  model_config.Register(po);

  po->Register("decoding-method", &decoding_method,
               "decoding method,"
               "Valid values: greedy_search.");
}

bool OfflineRecognizerConfig::Validate() const {
  return model_config.Validate();
}

std::string OfflineRecognizerConfig::ToString() const {
  std::ostringstream os;

  os << "OfflineRecognizerConfig(";
  os << "feat_config=" << feat_config.ToString() << ", ";
  os << "model_config=" << model_config.ToString() << ", ";
  os << "decoding_method=\"" << decoding_method << "\")";

  return os.str();
}

OfflineRecognizer::OfflineRecognizer(const OfflineRecognizerConfig &config)
    : impl_(OfflineRecognizerImpl::Create(config)) {}

OfflineRecognizer::~OfflineRecognizer() = default;

std::unique_ptr<OfflineStream> OfflineRecognizer::CreateStream() const {
  return impl_->CreateStream();
}

void OfflineRecognizer::DecodeStreams(OfflineStream **ss, int32_t n) const {
  impl_->DecodeStreams(ss, n);
}

}  // namespace sherpa_onnx
