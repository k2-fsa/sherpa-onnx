// sherpa-onnx/csrc/medibunny-phonemizer-frontend.h
//
// Medibunny fork — Copyright (c) 2026 Medibunny Ltd.
// Apache-2.0 License (same as sherpa-onnx upstream)
//
// Custom TTS frontend that receives pre-computed phoneme IDs from Medibunny's
// Rust G2P pipeline. Replaces espeak-ng (GPL-3.0) entirely.
//
// Design:
//   1. Rust code runs phonemization (goruut / epitran / neural ONNX tiers).
//   2. Rust calls SherpaOnnxMedibunnySetPhonemeIds() via C FFI with the IDs.
//   3. sherpa-onnx calls ConvertTextToTokenIds() — this returns the stored IDs.
//   4. The TTS engine (VITS, Kokoro, Matcha…) synthesises audio from those IDs.
//
// Thread safety: SetPhonemeIds / ConvertTextToTokenIds are protected by a mutex.
// The intended use is single-threaded per TTS synthesis call (set → synthesise).

#ifndef SHERPA_ONNX_CSRC_MEDIBUNNY_PHONEMIZER_FRONTEND_H_
#define SHERPA_ONNX_CSRC_MEDIBUNNY_PHONEMIZER_FRONTEND_H_

#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

#include "sherpa-onnx/csrc/offline-tts-frontend.h"

namespace sherpa_onnx {

// GPL-free TTS frontend for the Medibunny sherpa-onnx fork.
//
// Usage from Rust (via C API):
//
//   // 1. Phonemize text in Rust → phoneme IDs
//   let ids: Vec<i64> = g2p_pipeline.phonemize(text, lang)?;
//
//   // 2. Hand IDs to sherpa before synthesis
//   SherpaOnnxMedibunnySetPhonemeIds(tts_handle, ids.as_ptr(), ids.len() as i32);
//
//   // 3. Call normal generate — text is ignored by this frontend
//   let audio = SherpaOnnxOfflineTtsGenerate(tts_handle, text_cstr, sid, speed);
//
class MedibunnyPhonemizerFrontend : public OfflineTtsFrontend {
 public:
  MedibunnyPhonemizerFrontend() = default;
  ~MedibunnyPhonemizerFrontend() override = default;

  // Store pre-computed phoneme IDs produced by our Rust G2P pipeline.
  // Called from Rust via SherpaOnnxMedibunnySetPhonemeIds() before each
  // synthesis call.  The next ConvertTextToTokenIds() call will consume them.
  void SetPhonemeIds(const std::vector<int64_t>& ids);

  // OfflineTtsFrontend interface.
  //
  // The `text` and `voice` parameters are intentionally ignored — all
  // phonemization happens in Rust before this is called.  Returns a
  // single-element vector wrapping the stored phoneme ID sequence.
  //
  // Returns an empty outer vector only when no IDs have been set yet
  // (which is a programming error — Rust must call SetPhonemeIds first).
  std::vector<TokenIDs> ConvertTextToTokenIds(
      const std::string& text,
      const std::string& voice = "") const override;

 private:
  mutable std::mutex mutex_;
  std::vector<int64_t> phoneme_ids_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_MEDIBUNNY_PHONEMIZER_FRONTEND_H_
