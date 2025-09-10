// sherpa-onnx/csrc/offline-tts-piper-impl.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-tts-piper-impl.h"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/onnx-utils.h"
#include "sherpa-onnx/csrc/piper-phoneme-ids.h"
#include "sherpa-onnx/csrc/piper-phonemize.h"
#include "sherpa-onnx/csrc/piper-voice.h"
#include "sherpa-onnx/csrc/file-utils.h"

// espeak-ng integration
extern "C" {
#include "espeak-ng/speak_lib.h"
}

#include "phonemize.hpp"

namespace sherpa_onnx {
void CallPhonemizeEspeak(const std::string &text,
                         ::piper::eSpeakPhonemeConfig &config,
                         std::vector<std::vector<::piper::Phoneme>> *phonemes);

void InitEspeak(const std::string &data_dir);
}

namespace sherpa_onnx {

OfflineTtsPiperImpl::OfflineTtsPiperImpl(const OfflineTtsConfig &config)
    : config_(config),
      model_(std::make_unique<OfflineTtsPiperModel>(config.model)) {
  InitFrontend();
  
  // Load voice configuration from model files
  if (!LoadVoiceConfig()) {
    SHERPA_ONNX_LOGE("Failed to load voice configuration");
  }
}

#if __ANDROID_API__ >= 9
OfflineTtsPiperImpl::OfflineTtsPiperImpl(AAssetManager *mgr,
                                         const OfflineTtsConfig &config)
    : config_(config),
      model_(std::make_unique<OfflineTtsPiperModel>(mgr, config.model)) {
  InitFrontend();
  
  // Load voice configuration from model files
  if (!LoadVoiceConfig()) {
    SHERPA_ONNX_LOGE("Failed to load voice configuration");
  }
}
#endif

GeneratedAudio OfflineTtsPiperImpl::Generate(
    const std::string &text, int64_t sid /*= 0*/, float speed /*= 1.0*/,
    GeneratedAudioCallback callback /*= nullptr*/) const {
  
  GeneratedAudio result;
  
  if (text.empty()) {
    SHERPA_ONNX_LOGE("Empty text provided");
    return result;
  }

  // Initialize audio result
  result.sample_rate = SampleRate();
  
  // Calculate sentence silence samples
  std::size_t sentence_silence_samples = 0;
  const auto &meta_data = model_->GetMetaData();
  if (meta_data.sentence_silence_seconds > 0) {
    sentence_silence_samples = static_cast<std::size_t>(
        meta_data.sentence_silence_seconds * result.sample_rate);
  }

  if (config_.model.debug) {
    SHERPA_ONNX_LOGE("Phonemizing text: %s", text.c_str());
  }

  // Step 1: Text to phonemes using Piper's phonemize implementation  
  std::vector<std::vector<::piper::Phoneme>> phonemes;
  
  if (config_.model.debug) {
    SHERPA_ONNX_LOGE("Using voice from loaded config: %s", voice_data_.phonemizeConfig.voice.c_str());
  }
  
  // Phonemize text using loaded voice configuration
  if (!sherpa_onnx::piper::phonemize(text, voice_data_.phonemizeConfig, phonemes)) {
    SHERPA_ONNX_LOGE("Failed to phonemize text");
    return result;
  }

  if (config_.model.debug) {
    SHERPA_ONNX_LOGE("Phonemized text into %zu sentence(s)", phonemes.size());
  }

  // Step 2: Synthesize each sentence independently
  std::vector<sherpa_onnx::piper::PhonemeId> phoneme_ids;
  std::map<sherpa_onnx::piper::Phoneme, std::size_t> missing_phonemes;
  
  for (auto phonemes_iter = phonemes.begin(); phonemes_iter != phonemes.end(); ++phonemes_iter) {
    std::vector<::piper::Phoneme>& sentence_phonemes = *phonemes_iter;
    
    if (config_.model.debug && sentence_phonemes.size() > 0) {
      // DEBUG log for phonemes
      std::ostringstream phonemes_str;
      for (auto phoneme : sentence_phonemes) {
        if (phoneme < 128 && std::isprint(static_cast<char>(phoneme))) {
          phonemes_str << static_cast<char>(phoneme);
        } else {
          phonemes_str << "U+" << std::hex << static_cast<uint32_t>(phoneme) << " ";
        }
      }
      
      SHERPA_ONNX_LOGE("Converting %zu phoneme(s) to ids: %s", 
                       sentence_phonemes.size(), phonemes_str.str().c_str());
    }

    // Process phrases
    std::vector<std::shared_ptr<std::vector<::piper::Phoneme>>> phrase_phonemes;
    std::vector<std::size_t> phrase_silence_samples;
    
    // For now, treat entire sentence as one phrase
    phrase_phonemes.push_back(
        std::make_shared<std::vector<::piper::Phoneme>>(sentence_phonemes));
    phrase_silence_samples.push_back(0);

    // Use phoneme/id map from loaded voice config (like UE piper.cpp)
    sherpa_onnx::piper::PhonemeIdConfig id_config;
    id_config.phonemeIdMap = 
        std::make_shared<sherpa_onnx::piper::PhonemeIdMap>(voice_data_.phonemizeConfig.phonemeIdMap);

    std::vector<float> sentence_audio_buffer;

    // Process each phrase: phonemes -> ids -> audio 
    for (size_t phrase_idx = 0; phrase_idx < phrase_phonemes.size(); phrase_idx++) 
    {
      if (phrase_phonemes[phrase_idx]->size() <= 0) {
        continue;
      }

      // Convert phonemes to phoneme IDs
      phoneme_ids.clear();
      sherpa_onnx::piper::phonemes_to_ids(*phrase_phonemes[phrase_idx], id_config, phoneme_ids, missing_phonemes);
      
      if (config_.model.debug) {
        SHERPA_ONNX_LOGE("Converted %zu phoneme(s) to %zu phoneme ID(s) for phrase %zu",
                         phrase_phonemes[phrase_idx]->size(), phoneme_ids.size(), phrase_idx);
        
        std::ostringstream phoneme_ids_str;
        for (auto phoneme_id : phoneme_ids) {
          phoneme_ids_str << phoneme_id << ", ";
        }
        
        SHERPA_ONNX_LOGE("Phoneme IDs (%zu total): %s",
                         phoneme_ids.size(), phoneme_ids_str.str().c_str());
      }
     
      // ids -> audio
      sherpa_onnx::piper::SynthesisResult synthesis_result;
      
      // Create a simple callback that captures audio
      auto audio_callback = [&sentence_audio_buffer](std::vector<float>&& audio) {
        sentence_audio_buffer.insert(sentence_audio_buffer.end(), 
                                    audio.begin(), audio.end());
      };
      
      // Synthesize audio from phoneme IDs using Piper's synthesis pipeline
      if (!sherpa_onnx::piper::synthesize(phoneme_ids, voice_data_.synthesisConfig, 
                            voice_data_.session, sentence_audio_buffer, synthesis_result)) {
        SHERPA_ONNX_LOGE("Failed to synthesize audio from phoneme IDs");
        return result;
      }

      // Add end of phrase silence
      for (std::size_t i = 0; i < phrase_silence_samples[phrase_idx]; i++) {
        sentence_audio_buffer.push_back(0.0f);
      }
    }

    // Add sentence audio to result
    result.samples.insert(result.samples.end(), 
                         sentence_audio_buffer.begin(), 
                         sentence_audio_buffer.end());

    // Add end of sentence silence
    if (sentence_silence_samples > 0) {
      for (std::size_t i = 0; i < sentence_silence_samples; i++) {
        result.samples.push_back(0.0f);
      }
    }

    // Call callback with sentence audio 
    if (callback && !sentence_audio_buffer.empty()) {
      float progress = static_cast<float>(phonemes_iter - phonemes.begin() + 1) / phonemes.size();
      if (!callback(sentence_audio_buffer.data(), 
                   static_cast<int32_t>(sentence_audio_buffer.size()), 
                   progress)) {
        SHERPA_ONNX_LOGE("Synthesis cancelled by callback");
        return result;
      }
    }
  }

  // Report missing phonemes
  if (missing_phonemes.size() > 0) {
    SHERPA_ONNX_LOGE("Missing %zu phoneme(s) from phoneme/id map!", 
                     missing_phonemes.size());

    for (const auto& phoneme_count : missing_phonemes) {
      SHERPA_ONNX_LOGE("Missing \"U+%04X\": %zu time(s)", 
                       static_cast<uint32_t>(phoneme_count.first),
                       phoneme_count.second);
    }
  }

  return result;
}

int32_t OfflineTtsPiperImpl::SampleRate() const {
  return model_->GetMetaData().sample_rate;
}

int32_t OfflineTtsPiperImpl::NumSpeakers() const {
  return model_->GetMetaData().num_speakers;
}

void OfflineTtsPiperImpl::InitFrontend() {
  // Initialize espeak-ng if data directory is provided
  if (!config_.model.piper.data_dir.empty()) {
    InitEspeak(config_.model.piper.data_dir);
    
    if (config_.model.debug) {
      SHERPA_ONNX_LOGE("Initialized espeak-ng with data dir: %s", 
                       config_.model.piper.data_dir.c_str());
    }
  } else {
    SHERPA_ONNX_LOGE("Warning: No piper-data-dir provided. Phonemization may fail.");
  }
}

bool OfflineTtsPiperImpl::LoadVoiceConfig() {
  // Load Voice configuration
  
  if (config_.model.piper.model_config_file.empty()) {
    SHERPA_ONNX_LOGE("No model config file provided");
    return false;
  }
  
  // Read the config file
  std::vector<char> config_data = ReadFile(config_.model.piper.model_config_file);
  if (config_data.empty()) {
    SHERPA_ONNX_LOGE("Failed to read config file: %s", 
                     config_.model.piper.model_config_file.c_str());
    return false;
  }
  
  std::string config_content(config_data.begin(), config_data.end());
  
  // Read model data for loadVoice
  std::vector<char> model_data = ReadFile(config_.model.piper.model);
  if (model_data.empty()) {
    SHERPA_ONNX_LOGE("Failed to read model file: %s", 
                     config_.model.piper.model.c_str());
    return false;
  }
  
  if (!sherpa_onnx::piper::loadVoice(model_data.data(), model_data.size(), config_content, voice_data_)) {
    SHERPA_ONNX_LOGE("Failed to load voice configuration");
    return false;
  }
  
  if (config_.model.debug) {
    SHERPA_ONNX_LOGE("Successfully loaded voice configuration with %zu phoneme mappings",
                     voice_data_.phonemizeConfig.phonemeIdMap.size());
  }
  
  return true;
}


}  // namespace sherpa_onnx