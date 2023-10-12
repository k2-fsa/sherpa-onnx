// sherpa-onnx/csrc/online-recognizer-impl.h
//
// Copyright (c)  2023  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_ONLINE_RECOGNIZER_IMPL_H_
#define SHERPA_ONNX_CSRC_ONLINE_RECOGNIZER_IMPL_H_

#include <memory>
#include <string>
#include <vector>

#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/online-recognizer.h"
#include "sherpa-onnx/csrc/online-stream.h"

namespace sherpa_onnx {

class OnlineRecognizerImpl {
 public:
  static std::unique_ptr<OnlineRecognizerImpl> Create(
      const OnlineRecognizerConfig &config);

#if __ANDROID_API__ >= 9
  static std::unique_ptr<OnlineRecognizerImpl> Create(
      AAssetManager *mgr, const OnlineRecognizerConfig &config);
#endif

  virtual ~OnlineRecognizerImpl() = default;

  virtual std::unique_ptr<OnlineStream> CreateStream() const = 0;

  virtual std::unique_ptr<OnlineStream> CreateStream(
      const std::string &hotwords) const {
    SHERPA_ONNX_LOGE("Only transducer models support contextual biasing.");
    exit(-1);
  }

  virtual bool IsReady(OnlineStream *s) const = 0;

  virtual void DecodeStreams(OnlineStream **ss, int32_t n) const = 0;

  virtual OnlineRecognizerResult GetResult(OnlineStream *s) const = 0;

  virtual bool IsEndpoint(OnlineStream *s) const = 0;

  virtual void Reset(OnlineStream *s) const = 0;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_ONLINE_RECOGNIZER_IMPL_H_
