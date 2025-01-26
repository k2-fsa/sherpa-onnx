// sherpa-onnx/csrc/offline-tts-cache-mechanism-config.cc
//
// Copyright (c)  2025  @mah92 From Iranian people to the community with love

#include "sherpa-onnx/csrc/offline-tts-cache-mechanism-config.h"

#include <vector>

#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {

void OfflineTtsCacheMechanismConfig::Register(ParseOptions *po) {
  po->Register("tts-cache-dir", &cache_dir,
               "Path to the directory containing dict for espeak-ng.");
  po->Register("tts-cache-size", &cache_size,
               "Cache size for wav files in bytes. After the cache size is filled, wav files are kept based on usage statstics.");
}

bool OfflineTtsCacheMechanismConfig::Validate() const {
  return true;
}

std::string OfflineTtsCacheMechanismConfig::ToString() const {
  std::ostringstream os;

  os << "OfflineTtsCacheMechanismConfig(";
  os << "cache_dir=\"" << cache_dir << "\", ";
  os << "cache_size=" << cache_size << ")";

  return os.str();
}

}  // namespace sherpa_onnx
