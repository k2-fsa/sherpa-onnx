// sherpa-onnx/csrc/qwen3-asr-simulated-streaming-impl.h
//
// Copyright (c) 2026 Qing Peng
#ifndef SHERPA_ONNX_CSRC_QWEN3_ASR_SIMULATED_STREAMING_IMPL_H_
#define SHERPA_ONNX_CSRC_QWEN3_ASR_SIMULATED_STREAMING_IMPL_H_

#include <functional>
#include <memory>
#include <vector>

#include "sherpa-onnx/csrc/qwen3-asr-continuation.h"
#include "sherpa-onnx/csrc/qwen3-asr-simulated-streaming.h"

namespace sherpa_onnx {
// Private scheduler; the decoder is replaceable without loading model weights.
class Qwen3ASRStreamingState {
 public:
  using Decode = std::function<Qwen3ASRContinuationResult(
      const std::vector<float> &, const Qwen3ASRContinuationRequest &)>;
  Qwen3ASRStreamingState(const Qwen3ASRSimulatedStreamingConfig &config,
                         std::shared_ptr<QwenAsrTokenizer> tokenizer,
                         Decode decode);
  Qwen3ASRStreamingOutcome AcceptWaveform(const float *samples, int32_t n);
  Qwen3ASRStreamingOutcome Finish();
  void Reset();
  Qwen3ASRStreamingUpdate GetResult() const { return result_; }

 private:
  Qwen3ASRSimulatedStreamingConfig config_;
  std::shared_ptr<QwenAsrTokenizer> tokenizer_;
  Decode decode_;
  std::vector<float> pcm_;
  std::vector<int64_t> raw_ids_;
  int32_t processed_ = 0;
  int32_t successful_chunks_ = 0;
  bool finished_ = false;
  Qwen3ASRStreamingStatus failure_ = Qwen3ASRStreamingStatus::kOk;
  Qwen3ASRStreamingUpdate result_;
  Qwen3ASRStreamingOutcome Outcome(Qwen3ASRStreamingStatus status) const;
  bool DecodeAt(int32_t end, Qwen3ASRStreamingOutcome *out);
};
}  // namespace sherpa_onnx
#endif  // SHERPA_ONNX_CSRC_QWEN3_ASR_SIMULATED_STREAMING_IMPL_H_
