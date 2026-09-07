// sherpa-onnx/csrc/piper-phonemize-lexicon.h
//
// Copyright (c)  2022-2023  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_PIPER_PHONEMIZE_LEXICON_H_
#define SHERPA_ONNX_CSRC_PIPER_PHONEMIZE_LEXICON_H_

#include <string>
#include <unordered_map>
#include <vector>

#include "sherpa-onnx/csrc/offline-tts-frontend.h"
#include "sherpa-onnx/csrc/offline-tts-kitten-model-meta-data.h"
#include "sherpa-onnx/csrc/offline-tts-kokoro-model-meta-data.h"
#include "sherpa-onnx/csrc/offline-tts-matcha-model-meta-data.h"
#include "sherpa-onnx/csrc/offline-tts-vits-model-meta-data.h"

namespace sherpa_onnx {

class PiperPhonemizeLexicon : public OfflineTtsFrontend {
 public:
  PiperPhonemizeLexicon(const std::string &tokens, const std::string &data_dir,
                        const OfflineTtsVitsModelMetaData &vits_meta_data);

  PiperPhonemizeLexicon(const std::string &tokens, const std::string &data_dir,
                        const OfflineTtsMatchaModelMetaData &matcha_meta_data);

  PiperPhonemizeLexicon(const std::string &tokens, const std::string &data_dir,
                        const OfflineTtsKokoroModelMetaData &kokoro_meta_data);

  PiperPhonemizeLexicon(const std::string &tokens, const std::string &data_dir,
                        const OfflineTtsKittenModelMetaData &kitten_meta_data);

  template <typename Manager>
  PiperPhonemizeLexicon(Manager *mgr, const std::string &tokens,
                        const std::string &data_dir,
                        const OfflineTtsVitsModelMetaData &vits_meta_data);

  template <typename Manager>
  PiperPhonemizeLexicon(Manager *mgr, const std::string &tokens,
                        const std::string &data_dir,
                        const OfflineTtsMatchaModelMetaData &matcha_meta_data);

  template <typename Manager>
  PiperPhonemizeLexicon(Manager *mgr, const std::string &tokens,
                        const std::string &data_dir,
                        const OfflineTtsKokoroModelMetaData &kokoro_meta_data);

  template <typename Manager>
  PiperPhonemizeLexicon(Manager *mgr, const std::string &tokens,
                        const std::string &data_dir,
                        const OfflineTtsKittenModelMetaData &kitten_meta_data);

  std::vector<TokenIDs> ConvertTextToTokenIds(
      const std::string &text, const std::string &voice = "") const override;

 private:
  // map unicode codepoint to an integer ID; populated by every constructor
  // before it exits, preserving the original token-file validation.
  std::unordered_map<char32_t, int32_t> token2id_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_PIPER_PHONEMIZE_LEXICON_H_
