// sherpa-onnx/csrc/online-punctuation-model-config.cc
//
// Copyright (c) 2024 Jian You (jianyou@cisco.com, Cisco Systems)

#include "sherpa-onnx/csrc/online-punctuation-model-config.h"

#include <string>

#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {

void OnlinePunctuationModelConfig::Register(ParseOptions *po) {
  po->Register("cnn-bilstm", &cnn_bilstm,
               "Path to the light-weight CNN-BiLSTM model");

  po->Register("bpe-vocab", &bpe_vocab, "Path to the bpe vocab file");

  po->Register("num-threads", &num_threads,
               "Number of threads to run the neural network");

  po->Register("debug", &debug,
               "true to print model information while loading it.");

  po->Register("provider", &provider,
               "Specify a provider to use: cpu, cuda, coreml");
}

// 'exists' resolves a path either against the file system or against a
// resource manager.
template <typename Exists>
static bool ValidateOnlinePunctuationModelConfig(
    const OnlinePunctuationModelConfig &config, const Exists &exists) {
  if (config.cnn_bilstm.empty()) {
    SHERPA_ONNX_LOGE("Please provide --cnn-bilstm");
    return false;
  }

  if (!exists(config.cnn_bilstm)) {
    SHERPA_ONNX_LOGE("--cnn-bilstm '%s' does not exist",
                     config.cnn_bilstm.c_str());
    return false;
  }

  if (config.bpe_vocab.empty()) {
    SHERPA_ONNX_LOGE("Please provide --bpe-vocab");
    return false;
  }

  if (!exists(config.bpe_vocab)) {
    SHERPA_ONNX_LOGE("--bpe-vocab '%s' does not exist",
                     config.bpe_vocab.c_str());
    return false;
  }

  return true;
}

bool OnlinePunctuationModelConfig::Validate() const {
  return ValidateOnlinePunctuationModelConfig(
      *this, [](const std::string &p) -> bool { return FileExists(p); });
}

template <typename Manager>
bool OnlinePunctuationModelConfig::Validate(Manager *mgr) const {
  return ValidateOnlinePunctuationModelConfig(
      *this,
      [mgr](const std::string &p) -> bool { return FileExists(mgr, p); });
}

#if __ANDROID_API__ >= 9
template bool OnlinePunctuationModelConfig::Validate(AAssetManager *mgr) const;
#endif

#if __OHOS__
template bool OnlinePunctuationModelConfig::Validate(
    NativeResourceManager *mgr) const;
#endif

std::string OnlinePunctuationModelConfig::ToString() const {
  std::ostringstream os;

  os << "OnlinePunctuationModelConfig(";
  os << "cnn_bilstm=\"" << cnn_bilstm << "\", ";
  os << "bpe_vocab=\"" << bpe_vocab << "\", ";
  os << "num_threads=" << num_threads << ", ";
  os << "debug=" << (debug ? "True" : "False") << ", ";
  os << "provider=\"" << provider << "\")";

  return os.str();
}

}  // namespace sherpa_onnx
