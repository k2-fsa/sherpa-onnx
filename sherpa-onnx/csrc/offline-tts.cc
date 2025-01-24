// sherpa-onnx/csrc/offline-tts.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-tts.h"

#include <string>
#include <utility>

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#if __OHOS__
#include "rawfile/raw_file_manager.h"
#endif

#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/offline-tts-cache-mechanism.h"
#include "sherpa-onnx/csrc/offline-tts-impl.h"
#include "sherpa-onnx/csrc/text-utils.h"

namespace sherpa_onnx {

void OfflineTtsConfig::Register(ParseOptions *po) {
  model.Register(po);

  po->Register("tts-rule-fsts", &rule_fsts,
               "It not empty, it contains a list of rule FST filenames."
               "Multiple filenames are separated by a comma and they are "
               "applied from left to right. An example value: "
               "rule1.fst,rule2.fst,rule3.fst");

  po->Register("tts-rule-fars", &rule_fars,
               "It not empty, it contains a list of rule FST archive filenames."
               "Multiple filenames are separated by a comma and they are "
               "applied from left to right. An example value: "
               "rule1.far,rule2.far,rule3.far. Note that an *.far can contain "
               "multiple *.fst files");

  po->Register(
      "tts-max-num-sentences", &max_num_sentences,
      "Maximum number of sentences that we process at a time. "
      "This is to avoid OOM for very long input text. "
      "If you set it to -1, then we process all sentences in a single batch.");
}

bool OfflineTtsConfig::Validate() const {
  if (!rule_fsts.empty()) {
    std::vector<std::string> files;
    SplitStringToVector(rule_fsts, ",", false, &files);
    for (const auto &f : files) {
      if (!FileExists(f)) {
        SHERPA_ONNX_LOGE("Rule fst '%s' does not exist. ", f.c_str());
        return false;
      }
    }
  }

  if (!rule_fars.empty()) {
    std::vector<std::string> files;
    SplitStringToVector(rule_fars, ",", false, &files);
    for (const auto &f : files) {
      if (!FileExists(f)) {
        SHERPA_ONNX_LOGE("Rule far '%s' does not exist. ", f.c_str());
        return false;
      }
    }
  }

  return model.Validate();
}

std::string OfflineTtsConfig::ToString() const {
  std::ostringstream os;

  os << "OfflineTtsConfig(";
  os << "model=" << model.ToString() << ", ";
  os << "rule_fsts=\"" << rule_fsts << "\", ";
  os << "rule_fars=\"" << rule_fars << "\", ";
  os << "max_num_sentences=" << max_num_sentences << ")";

  return os.str();
}

OfflineTts::OfflineTts(const OfflineTtsConfig &config)
    : config_(config),
     impl_(OfflineTtsImpl::Create(config)),
     cache_mechanism_(nullptr) {}

template <typename Manager>
OfflineTts::OfflineTts(Manager *mgr, const OfflineTtsConfig &config)
    : config_(config),
     impl_(OfflineTtsImpl::Create(mgr, config)),
     cache_mechanism_(nullptr) {}

OfflineTts::~OfflineTts() = default;

GeneratedAudio OfflineTts::Generate(
    const std::string &text, int64_t sid /*=0*/, float speed /*= 1.0*/,
    GeneratedAudioCallback callback /*= nullptr*/) const {
  // Generate a hash for the text
  std::hash<std::string> hasher;
  std::string text_hash = std::to_string(hasher(text));
  // SHERPA_ONNX_LOGE("Generated text hash: %s", text_hash.c_str());

  // Check if the cache mechanism is active and if the audio is already cached
  if (cache_mechanism_) {
    int32_t sample_rate;
    std::vector<float> samples
      = cache_mechanism_->GetWavFile(text_hash, &sample_rate);

    if (!samples.empty()) {
      SHERPA_ONNX_LOGE("Returning cached audio for hash:%s", text_hash.c_str());

      // If a callback is provided, call it with the cached audio
      if (callback) {
        int32_t result
          = callback(samples.data(), samples.size(), 1.0f /* progress */);
        if (result == 0) {
          // If the callback returns 0, stop further processing
          SHERPA_ONNX_LOGE("Callback requested to stop processing.");
          return {samples, sample_rate};
        }
      }

      // Return the cached audio
      return {samples, sample_rate};
    }
  }

  // Generate the audio if not cached
  GeneratedAudio audio = impl_->Generate(text, sid, speed, std::move(callback));

  // Cache the generated audio if the cache mechanism is active
  if (cache_mechanism_) {
    cache_mechanism_->AddWavFile(text_hash, audio.samples, audio.sample_rate);
    // SHERPA_ONNX_LOGE("Cached audio for text hash: %s", text_hash.c_str());
  }

  return audio;
}

int32_t OfflineTts::SampleRate() const { return impl_->SampleRate(); }

int32_t OfflineTts::CacheSize() const {
  return cache_mechanism_ ? cache_mechanism_->GetCacheSize() : 0;
}

void OfflineTts::SetCacheSize(const int32_t cache_size) {
  if (cache_size > 0) {
    if (!cache_mechanism_) {
      // Initialize the cache mechanism if it hasn't been initialized yet
      cache_mechanism_ = std::make_unique<OfflineTtsCacheMechanism>(
        config_.cache_dir, cache_size);
    } else {
      // Update the cache size if the cache mechanism is already initialized
      cache_mechanism_->SetCacheSize(cache_size);
    }
  } else if (cache_mechanism_) {
    // If cache size is set to 0 or negative, destroy the cache mechanism
    cache_mechanism_.reset();
  }
}

void OfflineTts::ClearCache() {
  if (cache_mechanism_) {
    cache_mechanism_->ClearCache();
  }
}

int32_t OfflineTts::GetTotalUsedCacheSize() {
  if (cache_mechanism_) {
    return cache_mechanism_->GetTotalUsedCacheSize();
  }
  return -1;
}

int32_t OfflineTts::NumSpeakers() const { return impl_->NumSpeakers(); }

#if __ANDROID_API__ >= 9
template OfflineTts::OfflineTts(AAssetManager *mgr,
                                const OfflineTtsConfig &config);
#endif

#if __OHOS__
template OfflineTts::OfflineTts(NativeResourceManager *mgr,
                                const OfflineTtsConfig &config);
#endif

}  // namespace sherpa_onnx
