// sherpa-onnx/csrc/online-wenet-ctc-model-config.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/online-wenet-ctc-model-config.h"

#include <string>

#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {

void OnlineWenetCtcModelConfig::Register(ParseOptions *po) {
  po->Register("wenet-ctc-model", &model,
               "Path to CTC model.onnx from WeNet. Please see "
               "https://github.com/k2-fsa/sherpa-onnx/pull/425");
  po->Register("wenet-ctc-chunk-size", &chunk_size,
               "Chunk size after subsampling used for decoding.");
  po->Register("wenet-ctc-num-left-chunks", &num_left_chunks,
               "Number of left chunks after subsampling used for decoding.");
}

// 'exists' resolves a path either against the file system or against a
// resource manager.
template <typename Exists>
static bool ValidateOnlineWenetCtcModelConfig(
    const OnlineWenetCtcModelConfig &config, const Exists &exists) {
  if (!exists(config.model)) {
    SHERPA_ONNX_LOGE("WeNet CTC model '%s' does not exist",
                     config.model.c_str());
    return false;
  }

  if (config.chunk_size <= 0) {
    SHERPA_ONNX_LOGE(
        "Please specify a positive value for --wenet-ctc-chunk-size. Currently "
        "given: %d",
        config.chunk_size);
    return false;
  }

  if (config.num_left_chunks <= 0) {
    SHERPA_ONNX_LOGE(
        "Please specify a positive value for --wenet-ctc-num-left-chunks. "
        "Currently given: %d. Note that if you want to use -1, please consider "
        "using a non-streaming model.",
        config.num_left_chunks);
    return false;
  }

  return true;
}

bool OnlineWenetCtcModelConfig::Validate() const {
  return ValidateOnlineWenetCtcModelConfig(
      *this, [](const std::string &p) -> bool { return FileExists(p); });
}

template <typename Manager>
bool OnlineWenetCtcModelConfig::Validate(Manager *mgr) const {
  return ValidateOnlineWenetCtcModelConfig(
      *this,
      [mgr](const std::string &p) -> bool { return FileExists(mgr, p); });
}

#if __ANDROID_API__ >= 9
template bool OnlineWenetCtcModelConfig::Validate(AAssetManager *mgr) const;
#endif

#if __OHOS__
template bool OnlineWenetCtcModelConfig::Validate(
    NativeResourceManager *mgr) const;
#endif

std::string OnlineWenetCtcModelConfig::ToString() const {
  std::ostringstream os;

  os << "OnlineWenetCtcModelConfig(";
  os << "model=\"" << model << "\", ";
  os << "chunk_size=" << chunk_size << ", ";
  os << "num_left_chunks=" << num_left_chunks << ")";

  return os.str();
}

}  // namespace sherpa_onnx
