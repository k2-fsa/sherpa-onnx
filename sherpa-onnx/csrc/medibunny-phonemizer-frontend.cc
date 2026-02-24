// sherpa-onnx/csrc/medibunny-phonemizer-frontend.cc
//
// Medibunny fork — Copyright (c) 2026 Medibunny Ltd.
// Apache-2.0 License (same as sherpa-onnx upstream)

#include "sherpa-onnx/csrc/medibunny-phonemizer-frontend.h"

#include <mutex>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {

void MedibunnyPhonemizerFrontend::SetPhonemeIds(
    const std::vector<int64_t>& ids) {
  std::lock_guard<std::mutex> lock(mutex_);
  phoneme_ids_ = ids;
}

std::vector<TokenIDs> MedibunnyPhonemizerFrontend::ConvertTextToTokenIds(
    const std::string& /*text*/, const std::string& /*voice*/) const {
  std::lock_guard<std::mutex> lock(mutex_);

  if (phoneme_ids_.empty()) {
    SHERPA_ONNX_LOGE(
        "MedibunnyPhonemizerFrontend: ConvertTextToTokenIds called with no "
        "phoneme IDs set. Call SherpaOnnxMedibunnySetPhonemeIds() before "
        "SherpaOnnxOfflineTtsGenerate().");
    return {};
  }

  // Return the pre-computed phoneme IDs as a single sentence.
  // The TTS acoustic model (VITS, Kokoro, Matcha…) treats these as
  // token IDs directly — the mapping was done in Rust.
  std::vector<TokenIDs> result;
  result.emplace_back(phoneme_ids_);
  return result;
}

}  // namespace sherpa_onnx
