// sherpa-onnx/csrc/spoken-language-identification.cc
//
// Copyright (c)  2024  Xiaomi Corporation

#include "sherpa-onnx/csrc/spoken-language-identification.h"

#include <string>

#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/spoken-language-identification-impl.h"

namespace sherpa_onnx {

void SpokenLanguageIdentificationConfig::Register(ParseOptions *po) {
  po->Register("model", &model,
               "Path to encoder of a whisper multilingual model. Support only "
               "tiny, base, small, mediu, large.");
  po->Register("num-threads", &num_threads,
               "Number of threads to run the neural network");

  po->Register("debug", &debug,
               "true to print model information while loading it.");

  po->Register("provider", &provider,
               "Specify a provider to use: cpu, cuda, coreml");
}

bool SpokenLanguageIdentificationConfig::Validate() const {
  if (model.empty()) {
    SHERPA_ONNX_LOGE("Please provide --model");
    return false;
  }

  if (!FileExists(model)) {
    SHERPA_ONNX_LOGE("--model: %s does not exist", model.c_str());
    return false;
  }

  return true;
}

std::string SpokenLanguageIdentificationConfig::ToString() const {
  std::ostringstream os;

  os << "SpokenLanguageIdentificationConfig(";
  os << "model=\"" << model << "\", ";
  os << "num_threads=" << num_threads << ", ";
  os << "debug=" << (debug ? "True" : "False") << ", ";
  os << "provider=\"" << provider << "\")";

  return os.str();
}

SpokenLanguageIdentification::SpokenLanguageIdentification(
    const SpokenLanguageIdentificationConfig &config)
    : impl_(SpokenLanguageIdentificationImpl::Create(config)) {}

SpokenLanguageIdentification::~SpokenLanguageIdentification() = default;

std::unique_ptr<OnlineStream> SpokenLanguageIdentification::CreateStream()
    const {
  return impl_->CreateStream();
}

bool SpokenLanguageIdentification::IsReady(OnlineStream *s) const {
  return impl_->IsReady(s);
}

std::string SpokenLanguageIdentification::Compute(OnlineStream *s) const {
  return impl_->Compute(s);
}

}  // namespace sherpa_onnx
