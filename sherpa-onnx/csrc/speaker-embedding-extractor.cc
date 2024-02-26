// sherpa-onnx/csrc/speaker-embedding-extractor.cc
//
// Copyright (c)  2024  Xiaomi Corporation

#include "sherpa-onnx/csrc/speaker-embedding-extractor.h"

#include <vector>

#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/speaker-embedding-extractor-impl.h"

namespace sherpa_onnx {

void SpeakerEmbeddingExtractorConfig::Register(ParseOptions *po) {
  po->Register("model", &model, "Path to the speaker embedding model.");
  po->Register("num-threads", &num_threads,
               "Number of threads to run the neural network");

  po->Register("debug", &debug,
               "true to print model information while loading it.");

  po->Register("provider", &provider,
               "Specify a provider to use: cpu, cuda, coreml");
}

bool SpeakerEmbeddingExtractorConfig::Validate() const {
  if (model.empty()) {
    SHERPA_ONNX_LOGE("Please provide --model");
    return false;
  }

  if (!FileExists(model)) {
    SHERPA_ONNX_LOGE("--speaker-embedding-model: %s does not exist",
                     model.c_str());
    return false;
  }

  return true;
}

std::string SpeakerEmbeddingExtractorConfig::ToString() const {
  std::ostringstream os;

  os << "SpeakerEmbeddingExtractorConfig(";
  os << "model=\"" << model << "\", ";
  os << "num_threads=" << num_threads << ", ";
  os << "debug=" << (debug ? "True" : "False") << ", ";
  os << "provider=\"" << provider << "\")";

  return os.str();
}

SpeakerEmbeddingExtractor::SpeakerEmbeddingExtractor(
    const SpeakerEmbeddingExtractorConfig &config)
    : impl_(SpeakerEmbeddingExtractorImpl::Create(config)) {}

#if __ANDROID_API__ >= 9
SpeakerEmbeddingExtractor::SpeakerEmbeddingExtractor(
    AAssetManager *mgr, const SpeakerEmbeddingExtractorConfig &config)
    : impl_(SpeakerEmbeddingExtractorImpl::Create(mgr, config)) {}
#endif

SpeakerEmbeddingExtractor::~SpeakerEmbeddingExtractor() = default;

int32_t SpeakerEmbeddingExtractor::Dim() const { return impl_->Dim(); }

std::unique_ptr<OnlineStream> SpeakerEmbeddingExtractor::CreateStream() const {
  return impl_->CreateStream();
}

bool SpeakerEmbeddingExtractor::IsReady(OnlineStream *s) const {
  return impl_->IsReady(s);
}

std::vector<float> SpeakerEmbeddingExtractor::Compute(OnlineStream *s) const {
  return impl_->Compute(s);
}

}  // namespace sherpa_onnx
