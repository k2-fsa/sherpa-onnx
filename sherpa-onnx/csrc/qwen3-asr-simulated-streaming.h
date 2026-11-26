// sherpa-onnx/csrc/qwen3-asr-simulated-streaming.h
//
// Copyright (c) 2026 Qing Peng
#ifndef SHERPA_ONNX_CSRC_QWEN3_ASR_SIMULATED_STREAMING_H_
#define SHERPA_ONNX_CSRC_QWEN3_ASR_SIMULATED_STREAMING_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "sherpa-onnx/csrc/offline-recognizer.h"

namespace sherpa_onnx {
class OfflineRecognizerQwen3ASRImpl;
class Qwen3ASRStreamingState;

struct Qwen3ASRSimulatedStreamingConfig {
  int32_t chunk_size_samples = 32000;
  int32_t unfixed_chunk_num = 2;
  int32_t unfixed_token_num = 5;
  int32_t max_utterance_samples = 480000;
};

struct Qwen3ASRSimulatedStreamingOptions {
  std::string language;  // Empty means automatic language detection.
  std::optional<std::string> hotwords;  // Unset inherits, empty disables.
};

enum class Qwen3ASRStreamingStatus {
  kOk,
  kInvalidArgument,
  kInputFinished,
  kAudioLimit,
  kContextLimit,
  kTokenLimit,
  kRepetitionStopped,
  kDecodeError
};

struct Qwen3ASRStreamingUpdate {
  OfflineRecognitionResult result;
  bool is_final = false;
  int64_t audio_end_sample = 0;
  int64_t revision = 0;
  bool had_invalid_utf8 = false;
};

// Observational metrics for each attempted snapshot, including failed attempts.
// Never used by scheduling or rollback. No audio or raw token content.
struct Qwen3ASRStreamingStep {
  int64_t audio_end_sample = 0;
  int64_t prompt_tokens = 0;
  int64_t retained_tokens = 0;
  int64_t generated_tokens = 0;
  int64_t context_limit = 0;
  std::string stop_reason;
  double service_seconds = 0;
  double encoder_seconds = 0;
  double decoder_seconds = 0;
  int32_t decoder_calls = 0;
};

struct Qwen3ASRStreamingOutcome {
  Qwen3ASRStreamingStatus status = Qwen3ASRStreamingStatus::kOk;
  std::vector<Qwen3ASRStreamingUpdate> updates;
  int64_t accepted_samples = 0;
  int64_t processed_samples = 0;
  std::vector<Qwen3ASRStreamingStep> steps;
};

// Core C++ API. Serialize calls across streams sharing a recognizer.
// Each update is a complete, revisable hypothesis. Fixed 16 kHz mono [-1,1].
class Qwen3ASRSimulatedStreamingStream {
 public:
  ~Qwen3ASRSimulatedStreamingStream();
  Qwen3ASRStreamingOutcome AcceptWaveform(const float *samples, int32_t n);
  Qwen3ASRStreamingOutcome Finish();
  void Reset();
  Qwen3ASRStreamingUpdate GetResult() const;

 private:
  friend class Qwen3ASRSimulatedStreamingRecognizer;
  explicit Qwen3ASRSimulatedStreamingStream(
      std::unique_ptr<Qwen3ASRStreamingState> state);
  std::unique_ptr<Qwen3ASRStreamingState> state_;
};

class Qwen3ASRSimulatedStreamingRecognizer {
 public:
  // Invalid new configuration/options throw std::invalid_argument. Existing
  // fatal model-loading errors retain the core library's behavior.
  Qwen3ASRSimulatedStreamingRecognizer(
      const OfflineRecognizerConfig &model_config,
      const Qwen3ASRSimulatedStreamingConfig &config = {});
  std::unique_ptr<Qwen3ASRSimulatedStreamingStream> CreateStream(
      const Qwen3ASRSimulatedStreamingOptions &options = {}) const;

 private:
  Qwen3ASRSimulatedStreamingConfig config_;
  std::shared_ptr<OfflineRecognizerQwen3ASRImpl> engine_;
};

}  // namespace sherpa_onnx
#endif  // SHERPA_ONNX_CSRC_QWEN3_ASR_SIMULATED_STREAMING_H_
