// sherpa-onnx/csrc/offline-tts-piper-model-meta-data.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-tts-piper-model-meta-data.h"

#include <sstream>

namespace sherpa_onnx {

std::string OfflineTtsPiperModelMetaData::ToString() const {
  std::ostringstream os;
  
  os << "OfflineTtsPiperModelMetaData(";
  os << "sample_rate=" << sample_rate << ", ";
  os << "num_speakers=" << num_speakers << ", ";
  os << "noise_scale=" << noise_scale << ", ";
  os << "length_scale=" << length_scale << ", ";
  os << "noise_w=" << noise_w << ", ";
  os << "sentence_silence_seconds=" << sentence_silence_seconds << ", ";
  os << "pad_id=" << pad_id << ", ";
  os << "bos_id=" << bos_id << ", ";
  os << "eos_id=" << eos_id << ", ";
  os << "intersperse_pad=" << (intersperse_pad ? "True" : "False") << ", ";
  os << "phoneme_map_size=" << phoneme_id_map.size();
  os << ")";
  
  return os.str();
}

}  // namespace sherpa_onnx