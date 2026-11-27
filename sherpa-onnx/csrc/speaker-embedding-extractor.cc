// sherpa-onnx/csrc/speaker-embedding-extractor.cc
//
// Copyright (c)  2024  Xiaomi Corporation

#include "sherpa-onnx/csrc/speaker-embedding-extractor.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#if __OHOS__
#include "rawfile/raw_file_manager.h"
#endif

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

template <typename Exists>
static bool ValidateSpeakerModel(const std::string &model,
                                 const Exists &exists) {
  if (model.empty()) {
    SHERPA_ONNX_LOGE("Please provide a speaker embedding extractor model");
    return false;
  }

  if (!exists(model)) {
    SHERPA_ONNX_LOGE("speaker embedding extractor model: '%s' does not exist",
                     model.c_str());
    return false;
  }

  return true;
}

bool SpeakerEmbeddingExtractorConfig::Validate() const {
  return ValidateSpeakerModel(
      model, [](const std::string &path) { return FileExists(path); });
}

template <typename Manager>
bool SpeakerEmbeddingExtractorConfig::Validate(Manager *mgr) const {
  return ValidateSpeakerModel(
      model, [mgr](const std::string &path) { return FileExists(mgr, path); });
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
    : impl_(SpeakerEmbeddingExtractorImpl::Create(config)) {
  if (!impl_) {
    throw std::runtime_error(
        "Failed to create the speaker embedding extractor. Please check your "
        "model: " + config.model);
  }
}

template <typename Manager>
SpeakerEmbeddingExtractor::SpeakerEmbeddingExtractor(
    Manager *mgr, const SpeakerEmbeddingExtractorConfig &config)
    : impl_(SpeakerEmbeddingExtractorImpl::Create(mgr, config)) {
  if (!impl_) {
    throw std::runtime_error(
        "Failed to create the speaker embedding extractor. Please check your "
        "model: " + config.model);
  }
}

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

#if __ANDROID_API__ >= 9
template bool SpeakerEmbeddingExtractorConfig::Validate(
    AAssetManager *mgr) const;
template SpeakerEmbeddingExtractor::SpeakerEmbeddingExtractor(
    AAssetManager *mgr, const SpeakerEmbeddingExtractorConfig &config);
#endif

#if __OHOS__
template bool SpeakerEmbeddingExtractorConfig::Validate(
    NativeResourceManager *mgr) const;
template SpeakerEmbeddingExtractor::SpeakerEmbeddingExtractor(
    NativeResourceManager *mgr, const SpeakerEmbeddingExtractorConfig &config);
#endif

}  // namespace sherpa_onnx
