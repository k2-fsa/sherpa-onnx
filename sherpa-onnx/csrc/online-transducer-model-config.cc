// sherpa-onnx/csrc/online-transducer-model-config.cc
//
// Copyright (c)  2023  Xiaomi Corporation
#include "sherpa-onnx/csrc/online-transducer-model-config.h"

#include <sstream>

namespace sherpa_onnx {

std::string OnlineTransducerModelConfig::ToString() const {
  std::ostringstream os;

  os << "OnlineTransducerModelConfig(";
  os << "encoder_filename=\"" << encoder_filename << "\", ";
  os << "decoder_filename=\"" << decoder_filename << "\", ";
  os << "joiner_filename=\"" << joiner_filename << "\", ";
  os << "num_threads=" << num_threads << ", ";
  os << "debug=" << (debug ? "True" : "False") << ")";

  return os.str();
}

}  // namespace sherpa_onnx
