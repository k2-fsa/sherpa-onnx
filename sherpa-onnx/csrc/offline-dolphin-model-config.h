// sherpa-onnx/csrc/offline-dolphin-model-config.h
//
// Copyright (c)  2025  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_OFFLINE_DOLPHIN_MODEL_CONFIG_H_
#define SHERPA_ONNX_CSRC_OFFLINE_DOLPHIN_MODEL_CONFIG_H_

#include <string>

#include "sherpa-onnx/csrc/parse-options.h"

namespace sherpa_onnx {

struct OfflineDolphinModelConfig {
  // Path to the CTC branch (encoder + CTC head) of Dolphin.
  // Required only when the attention encoder/decoder pair is not provided.
  std::string model;

  // Path to encoder.onnx of the attention decoder branch.
  // Optional. Required when `decoder` is given.
  std::string encoder;

  // Path to decoder.onnx of the attention decoder branch.
  // Optional. When it is non-empty, `encoder` must also be provided and the
  // attention decoder is used for recognition instead of CTC greedy search,
  // enabling language/region control via `language`/`region`.
  std::string decoder;

  // Language code for the attention decoder, e.g., "zh", "en", "fil".
  // Optional. When empty, the language is predicted by the decoder.
  // Requires the attention decoder.
  std::string language;

  // Region code for the attention decoder, e.g., "CN", "PH", "US".
  // Optional. Requires `language` to be set. When `language` is set but
  // `region` is empty, the region is predicted by the decoder.
  // Requires the attention decoder.
  std::string region;

  OfflineDolphinModelConfig() = default;
  explicit OfflineDolphinModelConfig(const std::string &model) : model(model) {}

  OfflineDolphinModelConfig(const std::string &model,
                            const std::string &encoder,
                            const std::string &decoder,
                            const std::string &language,
                            const std::string &region)
      : model(model),
        encoder(encoder),
        decoder(decoder),
        language(language),
        region(region) {}

  void Register(ParseOptions *po);
  bool Validate() const;

  std::string ToString() const;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_DOLPHIN_MODEL_CONFIG_H_
