// sherpa-onnx/csrc/offline-recognizer-funasr-nano-impl.h

#ifndef SHERPA_ONNX_CSRC_OFFLINE_RECOGNIZER_FUNASR_NANO_IMPL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_RECOGNIZER_FUNASR_NANO_IMPL_H_

#include <algorithm>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/funasr-nano-tokenizer.h"
#include "sherpa-onnx/csrc/offline-funasr-nano-model.h"
#include "sherpa-onnx/csrc/offline-model-config.h"
#include "sherpa-onnx/csrc/offline-recognizer-impl.h"
#include "sherpa-onnx/csrc/offline-recognizer.h"
#include "sherpa-onnx/csrc/pad-sequence.h"

namespace sherpa_onnx {

class OfflineRecognizerFunASRNanoImpl : public OfflineRecognizerImpl {
 public:
  explicit OfflineRecognizerFunASRNanoImpl(
      const OfflineRecognizerConfig &config);

  template <typename Manager>
  OfflineRecognizerFunASRNanoImpl(Manager *mgr,
                                  const OfflineRecognizerConfig &config);

  std::unique_ptr<OfflineStream> CreateStream() const override;

  void DecodeStreams(OfflineStream **ss, int32_t n) const override;

  OfflineRecognizerConfig GetConfig() const override { return config_; }

 private:
  void InitFeatConfig();
  std::vector<float> ApplyLFR(const std::vector<float> &in) const;
  std::vector<int64_t> BuildSourceIds(
      const std::string &system_prompt, const std::string &user_prompt,
      int32_t audio_token_len, int32_t &fbank_beg_idx,
      int32_t &fake_token_len) const;
  int64_t SampleToken(const float *logits, int32_t vocab_size, int32_t step,
                     int64_t eos_token_id, int64_t im_end_token_id) const;
  int64_t SampleTokenFromLogitsFp16OrFp32(const void *logits, bool is_fp16,
                                          int32_t vocab_size) const;
  OfflineRecognitionResult GenerateText(
      Ort::Value encoder_out, const std::string &system_prompt,
      const std::string &user_prompt) const;

  OfflineRecognizerConfig config_;
  std::unique_ptr<OfflineFunASRNanoModel> model_;
  std::unique_ptr<FunASRNanoTokenizer> tokenizer_;
  mutable std::mt19937 rng_;
};

#if __ANDROID_API__ >= 9
template OfflineRecognizerFunASRNanoImpl::OfflineRecognizerFunASRNanoImpl(
    AAssetManager *mgr, const OfflineRecognizerConfig &config);
#endif

#if __OHOS__
template OfflineRecognizerFunASRNanoImpl::OfflineRecognizerFunASRNanoImpl(
    NativeResourceManager *mgr, const OfflineRecognizerConfig &config);
#endif

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_RECOGNIZER_FUNASR_NANO_IMPL_H_

