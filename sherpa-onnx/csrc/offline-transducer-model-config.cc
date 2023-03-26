// sherpa-onnx/csrc/offline-transducer-model-config.cc
//
// Copyright (c)  2023  Xiaomi Corporation
#include "sherpa-onnx/csrc/offline-transducer-model-config.h"

#include <sstream>

#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {

void OfflineTransducerModelConfig::Register(ParseOptions *po) {
  po->Register("encoder", &encoder_filename, "Path to encoder.onnx");
  po->Register("decoder", &decoder_filename, "Path to decoder.onnx");
  po->Register("joiner", &joiner_filename, "Path to joiner.onnx");
  po->Register("tokens", &tokens, "Path to tokens.txt");
  po->Register("num_threads", &num_threads,
               "Number of threads to run the neural network");

  po->Register("debug", &debug,
               "true to print model information while loading it.");
}

bool OfflineTransducerModelConfig::Validate() const {
  if (!FileExists(tokens)) {
    SHERPA_ONNX_LOGE("%s does not exist", tokens.c_str());
    return false;
  }

  if (!FileExists(encoder_filename)) {
    SHERPA_ONNX_LOGE("%s does not exist", encoder_filename.c_str());
    return false;
  }

  if (!FileExists(decoder_filename)) {
    SHERPA_ONNX_LOGE("%s does not exist", decoder_filename.c_str());
    return false;
  }

  if (!FileExists(joiner_filename)) {
    SHERPA_ONNX_LOGE("%s does not exist", joiner_filename.c_str());
    return false;
  }

  if (num_threads < 1) {
    SHERPA_ONNX_LOGE("num_threads should be > 0. Given %d", num_threads);
    return false;
  }

  return true;
}

std::string OfflineTransducerModelConfig::ToString() const {
  std::ostringstream os;

  os << "OfflineTransducerModelConfig(";
  os << "encoder_filename=\"" << encoder_filename << "\", ";
  os << "decoder_filename=\"" << decoder_filename << "\", ";
  os << "joiner_filename=\"" << joiner_filename << "\", ";
  os << "tokens=\"" << tokens << "\", ";
  os << "num_threads=" << num_threads << ", ";
  os << "debug=" << (debug ? "True" : "False") << ")";

  return os.str();
}

}  // namespace sherpa_onnx
