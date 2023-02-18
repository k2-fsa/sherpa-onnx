// sherpa/csrc/online-transducer-model-config.h
//
// Copyright (c)  2023  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_ONLINE_TRANSDUCER_MODEL_CONFIG_H_
#define SHERPA_ONNX_CSRC_ONLINE_TRANSDUCER_MODEL_CONFIG_H_

#include <string>

namespace sherpa_onnx {

struct OnlineTransducerModelConfig {
  std::string encoder_filename;
  std::string decoder_filename;
  std::string joiner_filename;
  int32_t num_threads;
  bool debug = false;

  std::string ToString() const;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_ONLINE_TRANSDUCER_MODEL_CONFIG_H_
