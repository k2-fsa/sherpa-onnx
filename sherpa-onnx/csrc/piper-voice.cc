// sherpa-onnx/csrc/piper-voice.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa-onnx/csrc/piper-voice.h"

#include <algorithm>
#include <chrono>
#include <sstream>

#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {
namespace piper {

// JSON parsing helper functions - parse phoneme_id_map and espeak config from JSON
bool parsePhonemizeConfig(const std::string& configData, PhonemizeConfig& phonemizeConfig) {
  phonemizeConfig.phonemeType = PhonemizerPhonemes;
  
  // Parse espeak voice from JSON config
  if (!parseEspeakVoiceFromJson(configData, phonemizeConfig.voice)) {
    SHERPA_ONNX_LOGE("Failed to parse espeak voice from config, using default");
    phonemizeConfig.voice = "en-us"; // Fallback default
  }
  
  // Parse phoneme_id_map from JSON config
  if (!parsePhonemeIdMapFromJson(configData, phonemizeConfig.phonemeIdMap)) {
    SHERPA_ONNX_LOGE("Failed to parse phoneme_id_map from config, using defaults");
    // Fallback to default phoneme ID map
    phonemizeConfig.phonemeIdMap = {
      {U'_', {0}}, {U'^', {1}}, {U'$', {2}}, {U' ', {3}},
      {U'!', {4}}, {U'\'', {5}}, {U'(', {6}}, {U')', {7}},
      {U',', {8}}, {U'-', {9}}, {U'.', {10}}, {U':', {11}},
      {U';', {12}}, {U'?', {13}}
    };
  }
  
  if (phonemizeConfig.phonemeIdMap.empty()) {
    SHERPA_ONNX_LOGE("Warning: Empty phoneme_id_map loaded from config");
    return false;
  }
  
  SHERPA_ONNX_LOGE("Loaded %zu phoneme mappings from config", phonemizeConfig.phonemeIdMap.size());
  SHERPA_ONNX_LOGE("Using espeak voice: %s", phonemizeConfig.voice.c_str());
  return true;
}

// Parse espeak voice configuration from JSON config
bool parseEspeakVoiceFromJson(const std::string& jsonContent, std::string& espeakVoice) {
  // Find "espeak" section in JSON
  auto espeak_pos = jsonContent.find("\"espeak\"");
  if (espeak_pos == std::string::npos) {
    SHERPA_ONNX_LOGE("No espeak section found in config");
    return false;
  }
  
  // Find the opening brace of espeak section
  auto brace_start = jsonContent.find("{", espeak_pos);
  if (brace_start == std::string::npos) {
    SHERPA_ONNX_LOGE("Invalid espeak section format - no opening brace");
    return false;
  }
  
  // Find matching closing brace for espeak section
  int brace_count = 1;
  size_t pos = brace_start + 1;
  size_t brace_end = std::string::npos;
  
  while (pos < jsonContent.size() && brace_count > 0) {
    if (jsonContent[pos] == '{') brace_count++;
    else if (jsonContent[pos] == '}') {
      brace_count--;
      if (brace_count == 0) {
        brace_end = pos;
        break;
      }
    }
    pos++;
  }
  
  if (brace_end == std::string::npos) {
    SHERPA_ONNX_LOGE("Invalid espeak section format - no closing brace");
    return false;
  }
  
  // Extract the espeak section content
  std::string espeak_content = jsonContent.substr(brace_start + 1, brace_end - brace_start - 1);
  
  // Find "voice" field in espeak section
  auto voice_pos = espeak_content.find("\"voice\"");
  if (voice_pos == std::string::npos) {
    SHERPA_ONNX_LOGE("No voice field found in espeak section");
    return false;
  }
  
  // Find the colon after "voice"
  auto colon_pos = espeak_content.find(":", voice_pos);
  if (colon_pos == std::string::npos) {
    SHERPA_ONNX_LOGE("Invalid voice field format - no colon");
    return false;
  }
  
  // Find the opening quote of voice value
  auto quote_start = espeak_content.find("\"", colon_pos);
  if (quote_start == std::string::npos) {
    SHERPA_ONNX_LOGE("Invalid voice field format - no opening quote");
    return false;
  }
  
  // Find the closing quote of voice value
  auto quote_end = espeak_content.find("\"", quote_start + 1);
  if (quote_end == std::string::npos) {
    SHERPA_ONNX_LOGE("Invalid voice field format - no closing quote");
    return false;
  }
  
  // Extract voice value
  espeakVoice = espeak_content.substr(quote_start + 1, quote_end - quote_start - 1);
  
  if (espeakVoice.empty()) {
    SHERPA_ONNX_LOGE("Empty espeak voice found in config");
    return false;
  }
  
  SHERPA_ONNX_LOGE("Parsed espeak voice from config: %s", espeakVoice.c_str());
  return true;
}

bool parsePhonemeIdMapFromJson(const std::string& jsonContent, PhonemeIdMap& phonemeIdMap) {
  // Find "phoneme_id_map" in JSON
  auto phoneme_map_pos = jsonContent.find("\"phoneme_id_map\"");
  if (phoneme_map_pos == std::string::npos) {
    SHERPA_ONNX_LOGE("No phoneme_id_map found in config");
    return false;
  }
  
  // Find the opening brace of phoneme_id_map
  auto brace_start = jsonContent.find("{", phoneme_map_pos);
  if (brace_start == std::string::npos) {
    SHERPA_ONNX_LOGE("Invalid phoneme_id_map format - no opening brace");
    return false;
  }
  
  // Find matching closing brace
  int brace_count = 1;
  size_t pos = brace_start + 1;
  size_t brace_end = std::string::npos;
  
  while (pos < jsonContent.size() && brace_count > 0) {
    if (jsonContent[pos] == '{') brace_count++;
    else if (jsonContent[pos] == '}') {
      brace_count--;
      if (brace_count == 0) {
        brace_end = pos;
        break;
      }
    }
    pos++;
  }
  
  if (brace_end == std::string::npos) {
    SHERPA_ONNX_LOGE("Invalid phoneme_id_map format - no closing brace");
    return false;
  }
  
  // Extract the phoneme map content
  std::string map_content = jsonContent.substr(brace_start + 1, brace_end - brace_start - 1);
  
  // Parse individual phoneme mappings with proper UTF-8 support
  // Format: "phoneme": [id1, id2, ...] or "phoneme": id
  size_t search_pos = 0;
  int parsed_count = 0;
  
  while (true) {
    // Find next phoneme key
    auto quote_start = map_content.find("\"", search_pos);
    if (quote_start == std::string::npos) break;
    
    auto quote_end = map_content.find("\"", quote_start + 1);
    if (quote_end == std::string::npos) break;
    
    // Extract phoneme string (may be UTF-8 encoded)
    std::string phoneme_str = map_content.substr(quote_start + 1, quote_end - quote_start - 1);
    
    // Find the colon after the phoneme key
    auto colon_pos = map_content.find(":", quote_end);
    if (colon_pos == std::string::npos) break;
    
    // Find the value (should be array format: [id])
    auto value_start = map_content.find_first_not_of(" \t\n\r", colon_pos + 1);
    if (value_start == std::string::npos) break;
    
    std::vector<PhonemeId> ids;
    
    if (map_content[value_start] == '[') {
      // Array format: [id1, id2, ...]
      auto bracket_end = map_content.find("]", value_start);
      if (bracket_end == std::string::npos) break;
      
      std::string ids_str = map_content.substr(value_start + 1, bracket_end - value_start - 1);
      
      // Parse comma-separated IDs
      std::istringstream iss(ids_str);
      std::string id_token;
      while (std::getline(iss, id_token, ',')) {
        // Remove whitespace
        id_token.erase(std::remove_if(id_token.begin(), id_token.end(), ::isspace), id_token.end());
        if (!id_token.empty()) {
          try {
            ids.push_back(std::stoll(id_token));
          } catch (const std::exception& e) {
            SHERPA_ONNX_LOGE("Failed to parse phoneme ID: %s", id_token.c_str());
          }
        }
      }
      
      search_pos = bracket_end + 1;
    } else {
      // Single ID format (fallback)
      auto value_end = map_content.find_first_of(",}", value_start);
      if (value_end == std::string::npos) value_end = map_content.size();
      
      std::string id_str = map_content.substr(value_start, value_end - value_start);
      // Remove whitespace
      id_str.erase(std::remove_if(id_str.begin(), id_str.end(), ::isspace), id_str.end());
      
      if (!id_str.empty()) {
        try {
          ids.push_back(std::stoll(id_str));
        } catch (const std::exception& e) {
          SHERPA_ONNX_LOGE("Failed to parse phoneme ID: %s", id_str.c_str());
        }
      }
      
      search_pos = value_end + 1;
    }
    
    // Convert phoneme string to char32_t (UTF-32) with proper UTF-8 decoding
    if (!ids.empty() && !phoneme_str.empty()) {
      char32_t phoneme_char = 0;
      
      // Convert UTF-8 string to UTF-32 codepoint
      if (phoneme_str.size() == 1 && (phoneme_str[0] & 0x80) == 0) {
        // Single ASCII byte
        phoneme_char = static_cast<char32_t>(phoneme_str[0]);
      } else {
        // Multi-byte UTF-8 sequence - decode first codepoint
        const uint8_t* utf8_str = reinterpret_cast<const uint8_t*>(phoneme_str.c_str());
        size_t utf8_len = phoneme_str.length();
        
        if (utf8_len >= 1) {
          if ((utf8_str[0] & 0x80) == 0) {
            // 1-byte UTF-8 (ASCII)
            phoneme_char = utf8_str[0];
          } else if ((utf8_str[0] & 0xE0) == 0xC0 && utf8_len >= 2) {
            // 2-byte UTF-8
            phoneme_char = ((utf8_str[0] & 0x1F) << 6) | (utf8_str[1] & 0x3F);
          } else if ((utf8_str[0] & 0xF0) == 0xE0 && utf8_len >= 3) {
            // 3-byte UTF-8
            phoneme_char = ((utf8_str[0] & 0x0F) << 12) | 
                          ((utf8_str[1] & 0x3F) << 6) | 
                          (utf8_str[2] & 0x3F);
          } else if ((utf8_str[0] & 0xF8) == 0xF0 && utf8_len >= 4) {
            // 4-byte UTF-8
            phoneme_char = ((utf8_str[0] & 0x07) << 18) | 
                          ((utf8_str[1] & 0x3F) << 12) | 
                          ((utf8_str[2] & 0x3F) << 6) | 
                          (utf8_str[3] & 0x3F);
          }
        }
      }
      
      if (phoneme_char != 0) {
        phonemeIdMap[phoneme_char] = ids;
        parsed_count++;
        
        // Debug log for special characters
        if (phoneme_char > 127) {
          SHERPA_ONNX_LOGE("Parsed Unicode phoneme U+%04X -> %lld", 
                          static_cast<uint32_t>(phoneme_char), ids[0]);
        }
      } else {
        SHERPA_ONNX_LOGE("Failed to decode UTF-8 phoneme: %s", phoneme_str.c_str());
      }
    }
  }
  SHERPA_ONNX_LOGE("Parsed %d phoneme mappings from JSON config", parsed_count);
  
  return parsed_count > 0;
}

bool parseSynthesisConfig(const std::string& configData, SynthesisConfig& synthesisConfig) {
  // Simple parsing for synthesis parameters
  // TODO: Replace with proper JSON parsing
  
  synthesisConfig.sampleRate = 22050;
  synthesisConfig.channels = 1;
  synthesisConfig.noiseScale = 0.667f;
  synthesisConfig.lengthScale = 1.0f;
  synthesisConfig.noiseW = 0.8f;
  synthesisConfig.sentenceSilenceSeconds = 0.0f;
  
  return true;
}

void parseModelConfig(const std::string& configData, ModelConfig& modelConfig) {
  // Simple parsing for model configuration
  // TODO: Replace with proper JSON parsing
  
  modelConfig.numSpeakers = 1;
}

void loadModel(void* modelPtr, size_t modelSize, ModelSession& session) {
  SHERPA_ONNX_LOGE("Loading ONNX model with %zu bytes", modelSize);
  
  auto start_time = std::chrono::steady_clock::now();
  session.onnx = std::make_unique<Ort::Session>(session.env, modelPtr, modelSize, session.options);
  auto end_time = std::chrono::steady_clock::now();
  
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
  SHERPA_ONNX_LOGE("Loaded ONNX model in %lld ms", duration.count());
}

bool loadVoice(void* modelPtr, size_t modelSize,
               const std::string& modelConfigData, Voice& voiceData) {
  
  voiceData.configRoot = modelConfigData;

  // Parse phonemize configuration from JSON
  if (!parsePhonemizeConfig(modelConfigData, voiceData.phonemizeConfig)) {
    return false;
  }
  
  if (!parseSynthesisConfig(modelConfigData, voiceData.synthesisConfig)) {
    return false;
  }
  
  parseModelConfig(modelConfigData, voiceData.modelConfig);

  if (voiceData.modelConfig.numSpeakers > 1) {
    voiceData.synthesisConfig.speakerId = 0;
  }

  loadModel(modelPtr, modelSize, voiceData.session);
  
  SHERPA_ONNX_LOGE("Voice loaded successfully with %zu phoneme mappings", 
                   voiceData.phonemizeConfig.phonemeIdMap.size());
  
  return true;
}

bool synthesize(const std::vector<PhonemeId>& phonemeIds,
               SynthesisConfig& synthesisConfig, ModelSession& session,
               std::vector<float>& audioBuffer, SynthesisResult& result) {
  
  SHERPA_ONNX_LOGE("Synthesizing audio for %zu phoneme ID(s)", phonemeIds.size());

  auto memoryInfo = Ort::MemoryInfo::CreateCpu(
      OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

  // Allocate input tensors
  std::vector<int64_t> phonemeIdLengths{static_cast<int64_t>(phonemeIds.size())};
  std::vector<float> scales{
      synthesisConfig.noiseScale,
      synthesisConfig.lengthScale,
      synthesisConfig.noiseW
  };

  std::vector<Ort::Value> inputTensors;
  
  // Input phoneme IDs tensor [1, seq_len]
  std::vector<int64_t> phonemeIdsShape{1, static_cast<int64_t>(phonemeIds.size())};
  inputTensors.push_back(Ort::Value::CreateTensor<int64_t>(
      memoryInfo, const_cast<int64_t*>(phonemeIds.data()), phonemeIds.size(), 
      phonemeIdsShape.data(), phonemeIdsShape.size()));

  // Input lengths tensor [1]
  std::vector<int64_t> phonemeIdLengthsShape{static_cast<int64_t>(phonemeIdLengths.size())};
  inputTensors.push_back(Ort::Value::CreateTensor<int64_t>(
      memoryInfo, phonemeIdLengths.data(), phonemeIdLengths.size(),
      phonemeIdLengthsShape.data(), phonemeIdLengthsShape.size()));

  // Scales tensor [3]
  std::vector<int64_t> scalesShape{static_cast<int64_t>(scales.size())};
  inputTensors.push_back(Ort::Value::CreateTensor<float>(
      memoryInfo, scales.data(), scales.size(),
      scalesShape.data(), scalesShape.size()));

  // Speaker ID tensor (if multi-speaker)
  std::vector<int64_t> speakerId{
      static_cast<int64_t>(synthesisConfig.speakerId.value_or(0))
  };
  std::vector<int64_t> speakerIdShape{static_cast<int64_t>(speakerId.size())};

  if (synthesisConfig.speakerId) {
    inputTensors.push_back(Ort::Value::CreateTensor<int64_t>(
        memoryInfo, speakerId.data(), speakerId.size(), 
        speakerIdShape.data(), speakerIdShape.size()));
  }

  // Input and output names
  std::vector<const char*> inputNames = {"input", "input_lengths", "scales", "sid"};
  std::vector<const char*> outputNames = {"output"};

  // Run inference
  auto startTime = std::chrono::steady_clock::now();
  auto outputTensors = session.onnx->Run(
      Ort::RunOptions{nullptr}, inputNames.data(), inputTensors.data(),
      inputTensors.size(), outputNames.data(), outputNames.size());
  auto endTime = std::chrono::steady_clock::now();

  if ((outputTensors.size() != 1) || (!outputTensors.front().IsTensor())) {
    SHERPA_ONNX_LOGE("Invalid output tensors");
    return false;
  }

  auto inferDuration = std::chrono::duration<double>(endTime - startTime);
  result.inferSeconds = inferDuration.count();

  // Extract audio data
  const float* audio = outputTensors.front().GetTensorData<float>();
  auto audioShape = outputTensors.front().GetTensorTypeAndShapeInfo().GetShape();
  int64_t audioCount = audioShape[audioShape.size() - 1];

  result.audioSeconds = static_cast<double>(audioCount) / static_cast<double>(synthesisConfig.sampleRate);
  result.realTimeFactor = 0.0;
  if (result.audioSeconds > 0) {
    result.realTimeFactor = result.inferSeconds / result.audioSeconds;
  }

  SHERPA_ONNX_LOGE("Synthesized %f second(s) of audio in %f second(s)",
                   result.audioSeconds, result.inferSeconds);

  // Copy audio data to buffer
  audioBuffer.assign(audio, audio + audioCount);

  return true;
}

bool textToAudio(Voice& voiceData, const std::string& text, SynthesisResult& result,
                std::function<void(std::vector<float>&&)> AudioCallback,
                std::function<bool()> CancellationCheck) {
  
  if (!AudioCallback) {
    SHERPA_ONNX_LOGE("No audio callback provided");
    return false;
  }

  std::size_t sentenceSilenceSamples = 0;
  if (voiceData.synthesisConfig.sentenceSilenceSeconds > 0) {
    sentenceSilenceSamples = static_cast<std::size_t>(
        voiceData.synthesisConfig.sentenceSilenceSeconds *
        voiceData.synthesisConfig.sampleRate * voiceData.synthesisConfig.channels);
  }

  SHERPA_ONNX_LOGE("Phonemizing text: %s", text.c_str());

  // Check for cancellation
  if (CancellationCheck && CancellationCheck()) {
    SHERPA_ONNX_LOGE("Synthesis cancelled before phonemization");
    return false;
  }

  std::vector<std::vector<Phoneme>> phonemes;
  if (voiceData.phonemizeConfig.phonemeType == PhonemizerPhonemes) {
    if (!phonemize(text, voiceData.phonemizeConfig, phonemes)) {
      SHERPA_ONNX_LOGE("Failed to phonemize text");
      return false;
    }
  } else {
    // Use UTF-8 codepoints as "phonemes"
    CodepointsPhonemeConfig codepointsConfig;
    phonemize_codepoints(text, codepointsConfig, phonemes);
  }

  // Check for cancellation after phonemization
  if (CancellationCheck && CancellationCheck()) {
    SHERPA_ONNX_LOGE("Synthesis cancelled after phonemization");
    return false;
  }

  // Synthesize each sentence independently.
  std::vector<PhonemeId> phonemeIds;
  std::map<Phoneme, std::size_t> missingPhonemes;
  
  for (auto phonemesIter = phonemes.begin(); phonemesIter != phonemes.end(); ++phonemesIter) {
    // Check for cancellation at each sentence
    if (CancellationCheck && CancellationCheck()) {
      SHERPA_ONNX_LOGE("Synthesis cancelled during sentence processing");
      return false;
    }

    std::vector<Phoneme>& sentencePhonemes = *phonemesIter;

    SHERPA_ONNX_LOGE("Converting %zu phoneme(s) to ids", sentencePhonemes.size());

    std::vector<std::shared_ptr<std::vector<Phoneme>>> phrasePhonemes;
    std::vector<SynthesisResult> phraseResults;
    std::vector<size_t> phraseSilenceSamples;

    // Use phoneme/id map from config
    PhonemeIdConfig idConfig;
    idConfig.phonemeIdMap = std::make_shared<PhonemeIdMap>(voiceData.phonemizeConfig.phonemeIdMap);

    if (voiceData.synthesisConfig.phonemeSilenceSeconds) {
      // Split into phrases
      std::map<Phoneme, float>& phonemeSilenceSeconds = *voiceData.synthesisConfig.phonemeSilenceSeconds;

      auto currentPhrasePhonemes = std::make_shared<std::vector<Phoneme>>();
      phrasePhonemes.push_back(currentPhrasePhonemes);

      for (auto sentencePhonemesIter = sentencePhonemes.begin();
           sentencePhonemesIter != sentencePhonemes.end(); ++sentencePhonemesIter) {
        Phoneme& currentPhoneme = *sentencePhonemesIter;
        currentPhrasePhonemes->push_back(currentPhoneme);

        if (phonemeSilenceSeconds.count(currentPhoneme) > 0) {
          // Split at phrase boundary
          phraseSilenceSamples.push_back(
              static_cast<std::size_t>(phonemeSilenceSeconds[currentPhoneme] *
                  voiceData.synthesisConfig.sampleRate *
                  voiceData.synthesisConfig.channels));

          currentPhrasePhonemes = std::make_shared<std::vector<Phoneme>>();
          phrasePhonemes.push_back(currentPhrasePhonemes);
        }
      }
    } else {
      // Use all phonemes
      phrasePhonemes.push_back(std::make_shared<std::vector<Phoneme>>(sentencePhonemes));
    }

    // Ensure results/samples are the same size
    while (phraseResults.size() < phrasePhonemes.size()) {
      phraseResults.emplace_back();
    }

    while (phraseSilenceSamples.size() < phrasePhonemes.size()) {
      phraseSilenceSamples.push_back(0);
    }

    std::vector<float> audioBuffer;

    // phonemes -> ids -> audio
    for (size_t phraseIdx = 0; phraseIdx < phrasePhonemes.size(); phraseIdx++) {
      // Check for cancellation at each phrase
      if (CancellationCheck && CancellationCheck()) {
        SHERPA_ONNX_LOGE("Synthesis cancelled during phrase processing");
        return false;
      }

      if (phrasePhonemes[phraseIdx]->size() <= 0) {
        continue;
      }

      // phonemes -> ids
      phonemes_to_ids(*(phrasePhonemes[phraseIdx]), idConfig, phonemeIds, missingPhonemes);
      
      SHERPA_ONNX_LOGE("Converted %zu phoneme(s) to %zu phoneme id(s)",
                       phrasePhonemes[phraseIdx]->size(), phonemeIds.size());

      // Check for cancellation before synthesis
      if (CancellationCheck && CancellationCheck()) {
        SHERPA_ONNX_LOGE("Synthesis cancelled before audio generation");
        return false;
      }

      // ids -> audio
      if (!synthesize(phonemeIds, voiceData.synthesisConfig, voiceData.session, 
                     audioBuffer, phraseResults[phraseIdx])) {
        return false;
      }

      // Add end of phrase silence
      for (std::size_t i = 0; i < phraseSilenceSamples[phraseIdx]; i++) {
        audioBuffer.push_back(0.0f);
      }

      result.audioSeconds += phraseResults[phraseIdx].audioSeconds;
      result.inferSeconds += phraseResults[phraseIdx].inferSeconds;

      phonemeIds.clear();
    }

    // Add end of sentence silence
    if (sentenceSilenceSamples > 0) {
      for (std::size_t i = 0; i < sentenceSilenceSamples; i++) {
        audioBuffer.push_back(0.0f);
      }
    }

    // Check for cancellation before sending audio
    if (CancellationCheck && CancellationCheck()) {
      SHERPA_ONNX_LOGE("Synthesis cancelled before sending audio chunk");
      return false;
    }

    SHERPA_ONNX_LOGE("Calling audio callback with %zu sample(s)", audioBuffer.size());
    AudioCallback(std::move(audioBuffer));
    phonemeIds.clear();
  }

  if (missingPhonemes.size() > 0) {
    SHERPA_ONNX_LOGE("Missing %zu phoneme(s) from phoneme/id map!", missingPhonemes.size());

    for (const auto& phonemeCount : missingPhonemes) {
      SHERPA_ONNX_LOGE("Missing \"U+%04X\": %zu time(s)",
                       static_cast<uint32_t>(phonemeCount.first), phonemeCount.second);
    }
  }

  if (result.audioSeconds > 0) {
    result.realTimeFactor = result.inferSeconds / result.audioSeconds;
  }

  return true;
}

}  // namespace piper
}  // namespace sherpa_onnx