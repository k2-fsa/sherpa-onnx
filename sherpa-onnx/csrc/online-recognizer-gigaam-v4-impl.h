// sherpa-onnx/csrc/online-recognizer-gigaam-v4-impl.h

#ifndef SHERPA_ONNX_CSRC_ONLINE_RECOGNIZER_GIGAAM_V4_IMPL_H_
#define SHERPA_ONNX_CSRC_ONLINE_RECOGNIZER_GIGAAM_V4_IMPL_H_

#include <memory>

namespace sherpa_onnx {

class OnlineRecognizerImpl;
struct OnlineRecognizerConfig;

std::unique_ptr<OnlineRecognizerImpl> CreateOnlineGigaAmV4(
    const OnlineRecognizerConfig &config);

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_ONLINE_RECOGNIZER_GIGAAM_V4_IMPL_H_
