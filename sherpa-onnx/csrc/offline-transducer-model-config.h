// sherpa-onnx/csrc/offline-transducer-model-config.h
//
// Copyright (c)  2023  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_OFFLINE_TRANSDUCER_MODEL_CONFIG_H_
#define SHERPA_ONNX_CSRC_OFFLINE_TRANSDUCER_MODEL_CONFIG_H_

#include <string>

#include "sherpa-onnx/csrc/parse-options.h"

namespace sherpa_onnx {

struct OfflineTransducerModelConfig {
  std::string encoder_filename;
  std::string decoder_filename;
  std::string joiner_filename;
  std::string ctc;
  std::string frame_reducer;
  std::string encoder_proj;
  std::string decoder_proj;

  OfflineTransducerModelConfig() = default;
  OfflineTransducerModelConfig(const std::string &encoder_filename,
                               const std::string &decoder_filename,
                               const std::string &joiner_filename,
                               const std::string &ctc="",
                               const std::string &frame_reducer="",
                               const std::string &encoder_proj="",
                               const std::string &decoder_proj="")
      : encoder_filename(encoder_filename),
        decoder_filename(decoder_filename),
        joiner_filename(joiner_filename),
        ctc(ctc), 
        frame_reducer(frame_reducer),
        encoder_proj(encoder_proj),
        decoder_proj(decoder_proj) {}

  void Register(ParseOptions *po);
  bool Validate() const;

  std::string ToString() const;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_TRANSDUCER_MODEL_CONFIG_H_
