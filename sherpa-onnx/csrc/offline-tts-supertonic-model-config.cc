// sherpa-onnx/csrc/offline-tts-supertonic-model-config.cc
//
// Copyright (c)  2026 zengyw
//
// This file is based on Supertonic TTS
// (https://github.com/Supertone-Inc/supertonic) which is licensed under MIT
// License (Copyright (c) 2025 Supertone Inc.)

#include "sherpa-onnx/csrc/offline-tts-supertonic-model-config.h"

#include <sstream>
#include <string>

#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {

void OfflineTtsSupertonicModelConfig::Register(ParseOptions *po) {
  po->Register("supertonic-duration-predictor", &duration_predictor,
               "Path to duration_predictor.onnx");
  po->Register("supertonic-text-encoder", &text_encoder,
               "Path to text_encoder.onnx");
  po->Register("supertonic-vector-estimator", &vector_estimator,
               "Path to vector_estimator.onnx");
  po->Register("supertonic-vocoder", &vocoder, "Path to vocoder.onnx");
  po->Register("supertonic-model-dir", &model_dir,
               "Path to Supertonic model directory (for config files: "
               "tts.json and unicode_indexer.json)");
  po->Register("supertonic-voice-style", &voice_style,
               "Path to Supertonic voice style JSON file");
  po->Register("supertonic-num-steps", &num_steps,
               "Number of denoising steps (default: 5)");
  po->Register("supertonic-speed", &speed,
               "Speech speed factor (default: 1.05)");
  po->Register("supertonic-max-len-korean", &max_len_korean,
               "Maximum text chunk length for Korean (default: 120)");
  po->Register("supertonic-max-len-other", &max_len_other,
               "Maximum text chunk length for other languages (default: 300)");
}

bool OfflineTtsSupertonicModelConfig::Validate() const {
  if (duration_predictor.empty()) {
    SHERPA_ONNX_LOGE("Please provide --supertonic-duration-predictor");
    return false;
  }
  std::string abs_dp = ResolveAbsolutePath(duration_predictor);
  if (!FileExists(abs_dp)) {
    SHERPA_ONNX_LOGE("duration_predictor file does not exist: '%s'",
                     duration_predictor.c_str());
    return false;
  }

  if (text_encoder.empty()) {
    SHERPA_ONNX_LOGE("Please provide --supertonic-text-encoder");
    return false;
  }
  std::string abs_te = ResolveAbsolutePath(text_encoder);
  if (!FileExists(abs_te)) {
    SHERPA_ONNX_LOGE("text_encoder file does not exist: '%s'",
                     text_encoder.c_str());
    return false;
  }

  if (vector_estimator.empty()) {
    SHERPA_ONNX_LOGE("Please provide --supertonic-vector-estimator");
    return false;
  }
  std::string abs_ve = ResolveAbsolutePath(vector_estimator);
  if (!FileExists(abs_ve)) {
    SHERPA_ONNX_LOGE("vector_estimator file does not exist: '%s'",
                     vector_estimator.c_str());
    return false;
  }

  if (vocoder.empty()) {
    SHERPA_ONNX_LOGE("Please provide --supertonic-vocoder");
    return false;
  }
  std::string abs_voc = ResolveAbsolutePath(vocoder);
  if (!FileExists(abs_voc)) {
    SHERPA_ONNX_LOGE("vocoder file does not exist: '%s'", vocoder.c_str());
    return false;
  }

  if (model_dir.empty()) {
    SHERPA_ONNX_LOGE("Please provide --supertonic-model-dir");
    return false;
  }
  std::string abs_model_dir = ResolveAbsolutePath(model_dir);
  const char *required_config_files[] = {"tts.json", "unicode_indexer.json"};
  for (const char *filename : required_config_files) {
    std::string filepath = abs_model_dir + "/" + filename;
    if (!FileExists(filepath)) {
      SHERPA_ONNX_LOGE("%s does not exist in '%s'", filename,
                       abs_model_dir.c_str());
      return false;
    }
  }
  if (voice_style.empty()) {
    SHERPA_ONNX_LOGE("Please provide --supertonic-voice-style");
    return false;
  }
  size_t start = 0;
  size_t pos;
  while ((pos = voice_style.find(',', start)) != std::string::npos) {
    std::string path = voice_style.substr(start, pos - start);
    while (!path.empty() && std::isspace(static_cast<unsigned char>(path[0])))
      path.erase(0, 1);
    while (!path.empty() &&
           std::isspace(static_cast<unsigned char>(path.back())))
      path.pop_back();
    if (!path.empty()) {
      std::string abs_path = ResolveAbsolutePath(path);
      if (!FileExists(abs_path)) {
        SHERPA_ONNX_LOGE("Voice style file does not exist: '%s'",
                         abs_path.c_str());
        return false;
      }
    }
    start = pos + 1;
  }
  if (start < voice_style.length()) {
    std::string path = voice_style.substr(start);
    while (!path.empty() && std::isspace(static_cast<unsigned char>(path[0])))
      path.erase(0, 1);
    while (!path.empty() &&
           std::isspace(static_cast<unsigned char>(path.back())))
      path.pop_back();
    if (!path.empty()) {
      std::string abs_path = ResolveAbsolutePath(path);
      if (!FileExists(abs_path)) {
        SHERPA_ONNX_LOGE("Voice style file does not exist: '%s'",
                         abs_path.c_str());
        return false;
      }
    }
  }
  if (num_steps < 1) {
    SHERPA_ONNX_LOGE("num_steps must be >= 1, given: %d", num_steps);
    return false;
  }
  if (speed <= 0) {
    SHERPA_ONNX_LOGE("speed must be > 0, given: %f", speed);
    return false;
  }
  if (max_len_korean < 1) {
    SHERPA_ONNX_LOGE("max_len_korean must be >= 1, given: %d", max_len_korean);
    return false;
  }
  if (max_len_other < 1) {
    SHERPA_ONNX_LOGE("max_len_other must be >= 1, given: %d", max_len_other);
    return false;
  }
  return true;
}

std::string OfflineTtsSupertonicModelConfig::ToString() const {
  std::ostringstream os;
  os << "OfflineTtsSupertonicModelConfig(";
  os << "duration_predictor=\"" << duration_predictor << "\", ";
  os << "text_encoder=\"" << text_encoder << "\", ";
  os << "vector_estimator=\"" << vector_estimator << "\", ";
  os << "vocoder=\"" << vocoder << "\", ";
  os << "model_dir=\"" << model_dir << "\", ";
  os << "voice_style=\"" << voice_style << "\", ";
  os << "num_steps=" << num_steps << ", ";
  os << "speed=" << speed << ", ";
  os << "max_len_korean=" << max_len_korean << ", ";
  os << "max_len_other=" << max_len_other << ")";
  return os.str();
}

}  // namespace sherpa_onnx
