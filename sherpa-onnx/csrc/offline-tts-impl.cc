// sherpa-onnx/csrc/offline-tts-impl.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-tts-impl.h"

#include <memory>
#include <vector>

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#if __OHOS__
#include "rawfile/raw_file_manager.h"
#endif

#include "sherpa-onnx/csrc/offline-tts-kokoro-impl.h"
#include "sherpa-onnx/csrc/offline-tts-matcha-impl.h"
#include "sherpa-onnx/csrc/offline-tts-vits-impl.h"

namespace sherpa_onnx {

std::vector<int64_t> OfflineTtsImpl::AddBlank(const std::vector<int64_t> &x,
                                              int32_t blank_id /*= 0*/) const {
  // we assume the blank ID is 0
  std::vector<int64_t> buffer(x.size() * 2 + 1, blank_id);
  int32_t i = 1;
  for (auto k : x) {
    buffer[i] = k;
    i += 2;
  }
  return buffer;
}

std::unique_ptr<OfflineTtsImpl> OfflineTtsImpl::Create(
    const OfflineTtsConfig &config, OfflineTtsCacheMechanism* cache) {
  cache_ = cache;
  if (!config.model.vits.model.empty()) {
    return std::make_unique<OfflineTtsVitsImpl>(config);
  } else if (!config.model.matcha.acoustic_model.empty()) {
    return std::make_unique<OfflineTtsMatchaImpl>(config);
  }

  return std::make_unique<OfflineTtsKokoroImpl>(config);
}

template <typename Manager>
std::unique_ptr<OfflineTtsImpl> OfflineTtsImpl::Create(
    Manager *mgr, const OfflineTtsConfig &config, OfflineTtsCacheMechanism* cache) {
  cache_ = cache;
  if (!config.model.vits.model.empty()) {
    return std::make_unique<OfflineTtsVitsImpl>(mgr, config);
  } else if (!config.model.matcha.acoustic_model.empty()) {
    return std::make_unique<OfflineTtsMatchaImpl>(mgr, config);
  }

  return std::make_unique<OfflineTtsKokoroImpl>(mgr, config);
}

#if __ANDROID_API__ >= 9
template std::unique_ptr<OfflineTtsImpl> OfflineTtsImpl::Create(
    AAssetManager *mgr, const OfflineTtsConfig &config, OfflineTtsCacheMechanism* cache);
#endif
  
#if __OHOS__
template std::unique_ptr<OfflineTtsImpl> OfflineTtsImpl::Create(
    NativeResourceManager *mgr, const OfflineTtsConfig &config, OfflineTtsCacheMechanism* cache);
#endif
    
GeneratedAudio OfflineTtsImpl::GenerateWitchCache(
    const std::string &text, int64_t sid, float speed,
    GeneratedAudioCallback callback) const
{
  // Generate a hash for the text
  std::hash<std::string> hasher;
  std::size_t text_hash = hasher(text);

  //In phones, long texts come from messages, websites and book which are usually not repeated. Repeated text comes from menus and settings which are usually short 
  bool text_is_long = text.length() > 50? true: false;

  // Check if the cache mechanism is active and if the audio is already cached
  if (cache_ && !text_is_long) {
    int32_t sample_rate;
    std::vector<float> samples
      = cache_->GetWavFile(text_hash, &sample_rate);
    
    if (!samples.empty()) {
      SHERPA_ONNX_LOGE("Returning cached audio for hash: %zu", text_hash);
    
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
  
  auto audio = Generate(text, sid, speed, callback);

  // Cache the generated audio if the cache mechanism is active
  if (cache_ && !text_is_long) {
    cache_->AddWavFile(text_hash, audio.samples, audio.sample_rate);
  }

  return audio;
}
  
}  // namespace sherpa_onnx
