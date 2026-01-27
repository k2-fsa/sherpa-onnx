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
  po->Register("supertonic-model-dir", &model_dir,
               "Path to Supertonic model directory");
  po->Register("supertonic-voice-style", &voice_style,
               "Path to Supertonic voice style JSON file");
  po->Register("supertonic-num-steps", &num_steps,
               "Number of denoising steps (default: 5)");
  po->Register("supertonic-speed", &speed,
               "Speech speed factor (default: 1.05)");
}

bool OfflineTtsSupertonicModelConfig::Validate() const {
  if (model_dir.empty()) {
    SHERPA_ONNX_LOGE("Please provide --supertonic-model-dir");
    return false;
  }
  std::string abs_model_dir = ResolveAbsolutePath(model_dir);
  const char *required_files[] = {"duration_predictor.onnx",
                                  "text_encoder.onnx",
                                  "vector_estimator.onnx",
                                  "vocoder.onnx",
                                  "tts.json",
                                  "unicode_indexer.json"};
  for (const char *filename : required_files) {
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
    while (!path.empty() && std::isspace(path[0])) path.erase(0, 1);
    while (!path.empty() && std::isspace(path.back())) path.pop_back();
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
    while (!path.empty() && std::isspace(path[0])) path.erase(0, 1);
    while (!path.empty() && std::isspace(path.back())) path.pop_back();
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
  return true;
}

std::string OfflineTtsSupertonicModelConfig::ToString() const {
  std::ostringstream os;
  os << "OfflineTtsSupertonicModelConfig(";
  os << "model_dir=\"" << model_dir << "\", ";
  os << "voice_style=\"" << voice_style << "\", ";
  os << "num_steps=" << num_steps << ", ";
  os << "speed=" << speed << ")";
  return os.str();
}

}  // namespace sherpa_onnx
