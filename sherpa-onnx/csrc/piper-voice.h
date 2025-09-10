// sherpa-onnx/csrc/piper-voice.h
//
// Copyright (c)  2025  Xiaomi Corporation
// Adapted from Piper TTS Voice structure

#ifndef SHERPA_ONNX_CSRC_PIPER_VOICE_H_
#define SHERPA_ONNX_CSRC_PIPER_VOICE_H_

#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "onnxruntime_cxx_api.h"
#include "sherpa-onnx/csrc/piper-phoneme-ids.h"
#include "sherpa-onnx/csrc/piper-phonemize.h"

namespace sherpa_onnx {
namespace piper {

// Forward declarations
struct ModelSession;
struct PhonemizeConfig;
struct SynthesisConfig;
struct ModelConfig;
struct SynthesisResult;

// ONNX Model Session (equivalent to UE plugin ModelSession)
struct ModelSession {
  Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "piper"};
  Ort::SessionOptions options{};
  std::unique_ptr<Ort::Session> onnx;
  
  ModelSession() {
    env.DisableTelemetryEvents();
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
    options.DisableCpuMemArena();
    options.DisableMemPattern(); 
    options.DisableProfiling();
  }
};

// Phonemization configuration (equivalent to UE plugin PhonemizeConfig)
enum PhonemeType {
  PhonemizerPhonemes = 0,
  TextPhonemes = 1
};

struct PhonemizeConfig {
  std::string voice = "en-us";
  PhonemeType phonemeType = PhonemizerPhonemes;
  
  // Phoneme ID mapping loaded from model config
  PhonemeIdMap phonemeIdMap;
  
  // Optional phoneme mapping for text preprocessing
  std::shared_ptr<PhonemeMap> phonemeMap;
  
  // Phoneme punctuation settings (like PiperPhonemeConfig)
  Phoneme period = U'.';      // CLAUSE_PERIOD
  Phoneme comma = U',';       // CLAUSE_COMMA
  Phoneme question = U'?';    // CLAUSE_QUESTION
  Phoneme exclamation = U'!'; // CLAUSE_EXCLAMATION
  Phoneme colon = U':';       // CLAUSE_COLON
  Phoneme semicolon = U';';   // CLAUSE_SEMICOLON
  Phoneme space = U' ';
  
  // Remove language switch flags like "(en)"
  bool keepLanguageFlags = false;
};

// Synthesis configuration (equivalent to UE plugin SynthesisConfig)
struct SynthesisConfig {
  int32_t sampleRate = 22050;
  int32_t channels = 1;
  
  // Generation parameters
  float noiseScale = 0.667f;
  float lengthScale = 1.0f;
  float noiseW = 0.8f;
  
  // Silence configuration
  float sentenceSilenceSeconds = 0.0f;
  
  // Speaker configuration
  std::optional<int64_t> speakerId;
  
  // Phoneme silence mapping (phoneme -> seconds of silence to add after)
  std::optional<std::map<Phoneme, float>> phonemeSilenceSeconds;
};

// Model configuration (equivalent to UE plugin ModelConfig)
struct ModelConfig {
  int32_t numSpeakers = 1;
  
  // Optional speaker name to ID mapping
  std::optional<std::map<std::string, int64_t>> speakerIdMap;
};

// Synthesis result (equivalent to UE plugin SynthesisResult)
struct SynthesisResult {
  double audioSeconds = 0.0;
  double inferSeconds = 0.0;
  double realTimeFactor = 0.0;
};

// Complete Voice configuration (equivalent to UE plugin Voice)
struct Voice {
  // Model session for ONNX inference
  ModelSession session;
  
  // Configuration structures
  PhonemizeConfig phonemizeConfig;
  SynthesisConfig synthesisConfig;
  ModelConfig modelConfig;
  
  // JSON configuration root (for parsing)
  std::string configRoot;
  
  Voice() = default;
  ~Voice() = default;
  
  // Non-copyable but movable
  Voice(const Voice&) = delete;
  Voice& operator=(const Voice&) = delete;
  Voice(Voice&&) = default;
  Voice& operator=(Voice&&) = default;
};

// Function declarations (equivalent to UE plugin functions)

// Load voice from model data and config
bool loadVoice(void* modelPtr, size_t modelSize,
               const std::string& modelConfigData, Voice& voiceData);

// Main text to audio synthesis function (equivalent to UE plugin textToAudio)
bool textToAudio(Voice& voiceData, const std::string& text, SynthesisResult& result,
                std::function<void(std::vector<float>&&)> AudioCallback,
                std::function<bool()> CancellationCheck = nullptr);

// Phoneme IDs to WAV audio synthesis
bool synthesize(const std::vector<PhonemeId>& phonemeIds,
               SynthesisConfig& synthesisConfig, ModelSession& session,
               std::vector<float>& audioBuffer, SynthesisResult& result);

// JSON parsing helper functions
bool parsePhonemeIdMapFromJson(const std::string& jsonContent, PhonemeIdMap& phonemeIdMap);
bool parseEspeakVoiceFromJson(const std::string& jsonContent, std::string& espeakVoice);

}  // namespace piper
}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_PIPER_VOICE_H_