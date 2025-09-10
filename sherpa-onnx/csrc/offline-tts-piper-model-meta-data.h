// sherpa-onnx/csrc/offline-tts-piper-model-meta-data.h
//
// Copyright (c)  2025  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_OFFLINE_TTS_PIPER_MODEL_META_DATA_H_
#define SHERPA_ONNX_CSRC_OFFLINE_TTS_PIPER_MODEL_META_DATA_H_

#include <cstdint>
#include <string>
#include <unordered_map>

namespace sherpa_onnx {

struct OfflineTtsPiperModelMetaData {
  int32_t sample_rate = 22050;
  int32_t num_speakers = 1;
  
  // Phoneme configuration
  bool intersperse_pad = true;
  int64_t pad_id = 0;
  int64_t bos_id = 1;
  int64_t eos_id = 2;
  
  // Inference parameters
  float noise_scale = 0.667f;
  float length_scale = 1.0f;
  float noise_w = 0.8f;
  
  // Silence configuration (similar to UE plugin synthesisConfig)
  float sentence_silence_seconds = 0.0f;
  
  // Phoneme ID mapping
  std::unordered_map<char32_t, int64_t> phoneme_id_map;
  
  OfflineTtsPiperModelMetaData() = default;
  
  std::string ToString() const;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_TTS_PIPER_MODEL_META_DATA_H_