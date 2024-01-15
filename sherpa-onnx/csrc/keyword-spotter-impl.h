// sherpa-onnx/csrc/keyword-spotter-impl.h
//
// Copyright (c)  2023-2024  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_KEYWORD_SPOTTER_IMPL_H_
#define SHERPA_ONNX_CSRC_KEYWORD_SPOTTER_IMPL_H_

#include <memory>
#include <string>
#include <vector>

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#include "sherpa-onnx/csrc/keyword-spotter.h"
#include "sherpa-onnx/csrc/online-stream.h"

namespace sherpa_onnx {

class KeywordSpotterImpl {
 public:
  static std::unique_ptr<KeywordSpotterImpl> Create(
      const KeywordSpotterConfig &config);

#if __ANDROID_API__ >= 9
  static std::unique_ptr<KeywordSpotterImpl> Create(
      AAssetManager *mgr, const KeywordSpotterConfig &config);
#endif

  virtual ~KeywordSpotterImpl() = default;

  virtual std::unique_ptr<OnlineStream> CreateStream() const = 0;

  virtual std::unique_ptr<OnlineStream> CreateStream(
      const std::string &keywords) const = 0;

  virtual bool IsReady(OnlineStream *s) const = 0;

  virtual void DecodeStreams(OnlineStream **ss, int32_t n) const = 0;

  virtual KeywordResult GetResult(OnlineStream *s) const = 0;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_KEYWORD_SPOTTER_IMPL_H_
