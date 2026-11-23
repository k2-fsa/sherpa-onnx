// sherpa-onnx/csrc/qwen3-asr-simulated-streaming.cc
//
// Copyright (c) 2026 Qing Peng
#include "sherpa-onnx/csrc/qwen3-asr-simulated-streaming.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/offline-recognizer-qwen3-asr-impl.h"
#include "sherpa-onnx/csrc/qwen-asr-tokenizer.h"
#include "sherpa-onnx/csrc/qwen3-asr-continuation.h"
#include "sherpa-onnx/csrc/qwen3-asr-simulated-streaming-impl.h"
#include "utf8proc.h"  // NOLINT

namespace sherpa_onnx {
namespace {
bool IsValidUtf8(const std::string &bytes) {
  size_t i = 0;
  while (i < bytes.size()) {
    utf8proc_int32_t codepoint = 0;
    auto n = utf8proc_iterate(
        reinterpret_cast<const utf8proc_uint8_t *>(bytes.data() + i),
        bytes.size() - i, &codepoint);
    if (n <= 0) return false;
    i += n;
  }
  return true;
}

std::string NextDisplayCharacter(const std::string &bytes, size_t *offset) {
  utf8proc_int32_t codepoint = 0;
  auto n = utf8proc_iterate(
      reinterpret_cast<const utf8proc_uint8_t *>(bytes.data() + *offset),
      bytes.size() - *offset, &codepoint);
  if (n <= 0) {
    ++*offset;
    return "\xef\xbf\xbd";
  }
  auto character = bytes.substr(*offset, n);
  *offset += n;
  return character;
}

void ValidateConfig(const Qwen3ASRSimulatedStreamingConfig &c) {
  if (c.chunk_size_samples <= 0 || c.unfixed_chunk_num < 0 ||
      c.unfixed_token_num < 0 || c.max_utterance_samples <= 0 ||
      c.max_utterance_samples > 480000 ||
      c.chunk_size_samples > c.max_utterance_samples) {
    throw std::invalid_argument("Invalid Qwen3-ASR streaming limits");
  }
}

void ValidateHotwords(const std::string &text) {
  if (text.size() > 16384 || !IsValidUtf8(text) ||
      text.find("<|") != std::string::npos ||
      text.find("<asr_text>") != std::string::npos ||
      std::any_of(text.begin(), text.end(),
                  [](unsigned char c) { return c < 32 || c == 127; })) {
    throw std::invalid_argument("Invalid Qwen3-ASR hotwords");
  }
}

void ValidateOptions(const Qwen3ASRSimulatedStreamingOptions &options) {
  // Canonical names used by Qwen's language prompt (empty means automatic).
  static const std::vector<std::string> languages = {
      "",          "Chinese", "English", "Cantonese",  "Arabic",
      "German",    "French",  "Spanish", "Portuguese", "Indonesian",
      "Italian",   "Korean",  "Russian", "Thai",       "Vietnamese",
      "Japanese",  "Turkish", "Hindi",   "Malay",      "Dutch",
      "Swedish",   "Danish",  "Finnish", "Polish",     "Czech",
      "Filipino",  "Persian", "Greek",   "Romanian",   "Hungarian",
      "Macedonian"};
  if (std::find(languages.begin(), languages.end(), options.language) ==
      languages.end()) {
    throw std::invalid_argument("Invalid Qwen3-ASR language");
  }
  if (options.hotwords) ValidateHotwords(*options.hotwords);
}

Qwen3ASRStreamingStatus StatusFor(Qwen3ASRStopReason reason) {
  using Reason = Qwen3ASRStopReason;
  using Status = Qwen3ASRStreamingStatus;
  switch (reason) {
    case Reason::kEos:
    case Reason::kNoSpeech:
    case Reason::kInsufficientFrames:
      return Status::kOk;
    case Reason::kContextLimit:
      return Status::kContextLimit;
    case Reason::kTokenLimit:
      return Status::kTokenLimit;
    case Reason::kRepetitionStopped:
      return Status::kRepetitionStopped;
    default:
      return Status::kDecodeError;
  }
}
}  // namespace

std::vector<int64_t> MakeQwen3ASRPrefix(const std::vector<int64_t> &raw_ids,
                                        int32_t chunk_index,
                                        int32_t warmup_chunks,
                                        int32_t rollback_tokens,
                                        QwenAsrTokenizer &tokenizer) {
  if (chunk_index < 0 || warmup_chunks < 0 || rollback_tokens < 0) {
    throw std::invalid_argument("Invalid Qwen3-ASR rollback parameters");
  }
  if (chunk_index < warmup_chunks ||
      raw_ids.size() <= static_cast<size_t>(rollback_tokens))
    return {};
  std::vector<int64_t> prefix(raw_ids.begin(), raw_ids.end() - rollback_tokens);
  while (!prefix.empty() && !IsValidUtf8(tokenizer.Decode(prefix))) {
    prefix.pop_back();
  }
  return prefix;
}

OfflineRecognitionResult FormatQwen3ASRContinuation(
    const std::vector<int64_t> &raw_ids, const std::string &language,
    QwenAsrTokenizer &tokenizer, bool *had_invalid_utf8) {
  OfflineRecognitionResult result;
  const std::string full = tokenizer.Decode(raw_ids);
  *had_invalid_utf8 = !IsValidUtf8(full);
  result.lang = language;
  size_t first = 0;
  if (language.empty() && !full.empty()) {
    const std::string scaffold = "language ";
    if (full.compare(0, scaffold.size(), scaffold) == 0 ||
        scaffold.compare(0, full.size(), full) == 0) {
      auto marker = std::find(raw_ids.begin(), raw_ids.end(),
                              tokenizer.GetTokenId("<asr_text>"));
      if (marker == raw_ids.end()) return result;
      first = std::distance(raw_ids.begin(), marker) + 1;
      const auto end = full.find("<asr_text>");
      result.lang = full.substr(scaffold.size(), end - scaffold.size());
    }
  }
  std::string display_language;
  for (size_t i = 0; i < result.lang.size();) {
    display_language += NextDisplayCharacter(result.lang, &i);
  }
  result.lang = std::move(display_language);

  // Preserve token slots, assigning a split UTF-8 character to the token that
  // completes it. One replacement character per invalid byte, never dropped.
  std::string bytes;
  std::vector<size_t> ends;
  for (size_t i = first; i < raw_ids.size(); ++i) {
    bytes += tokenizer.Decode({raw_ids[i]});
    ends.push_back(bytes.size());
  }
  result.tokens.resize(ends.size());
  size_t slot = 0;
  for (size_t i = 0; i < bytes.size();) {
    auto character = NextDisplayCharacter(bytes, &i);
    while (slot + 1 < ends.size() && ends[slot] < i) ++slot;
    result.tokens[slot] += character;
    result.text += character;
  }
  return result;
}
Qwen3ASRStreamingState::Qwen3ASRStreamingState(
    const Qwen3ASRSimulatedStreamingConfig &config,
    std::shared_ptr<QwenAsrTokenizer> tokenizer, Decode decode)
    : config_(config),
      tokenizer_(std::move(tokenizer)),
      decode_(std::move(decode)) {
  ValidateConfig(config_);
  if (!tokenizer_ || !decode_) {
    throw std::invalid_argument("Missing Qwen3-ASR decoder");
  }
}

Qwen3ASRStreamingOutcome Qwen3ASRStreamingState::Outcome(
    Qwen3ASRStreamingStatus status) const {
  Qwen3ASRStreamingOutcome out;
  out.status = status;
  out.accepted_samples = pcm_.size();
  out.processed_samples = processed_;
  return out;
}

bool Qwen3ASRStreamingState::DecodeAt(int32_t end,
                                      Qwen3ASRStreamingOutcome *out) {
  const auto started = std::chrono::steady_clock::now();
  const size_t step_index = out->steps.size();
  try {
    out->steps.emplace_back();
    auto &step = out->steps.back();
    step.audio_end_sample = end;
    Qwen3ASRContinuationRequest request;
    request.strict_continuation = true;
    request.prefix_ids = MakeQwen3ASRPrefix(
        raw_ids_, successful_chunks_, config_.unfixed_chunk_num,
        config_.unfixed_token_num, *tokenizer_);
    step.retained_tokens = request.prefix_ids.size();
    std::vector<float> snapshot(pcm_.begin(), pcm_.begin() + end);
    auto decoded = decode_(snapshot, request);
    step.prompt_tokens = decoded.prompt_tokens;
    step.generated_tokens = decoded.generated_tokens;
    step.context_limit = decoded.context_limit;
    step.encoder_seconds = decoded.encoder_seconds;
    step.decoder_seconds = decoded.decoder_seconds;
    step.decoder_calls = decoded.decoder_calls;
    switch (decoded.stop_reason) {
      case Qwen3ASRStopReason::kEos:
        step.stop_reason = "eos";
        break;
      case Qwen3ASRStopReason::kNoSpeech:
        step.stop_reason = "no_speech";
        break;
      case Qwen3ASRStopReason::kInsufficientFrames:
        step.stop_reason = "insufficient_frames";
        break;
      case Qwen3ASRStopReason::kContextLimit:
        step.stop_reason = "context_limit";
        break;
      case Qwen3ASRStopReason::kTokenLimit:
        step.stop_reason = "token_limit";
        break;
      case Qwen3ASRStopReason::kRepetitionStopped:
        step.stop_reason = "repetition_stopped";
        break;
      default:
        step.stop_reason = "decode_error";
    }
    failure_ = StatusFor(decoded.stop_reason);
    if (failure_ == Qwen3ASRStreamingStatus::kOk) {
      Qwen3ASRStreamingUpdate update;
      update.result = std::move(decoded.result);
      update.audio_end_sample = end;
      update.revision = result_.revision + 1;
      update.had_invalid_utf8 = decoded.had_invalid_utf8;
      // Allocate the outcome before committing any successful state.
      out->updates.push_back(update);
      result_ = std::move(update);
      raw_ids_ = std::move(decoded.raw_output_ids);
      processed_ = end;
      ++successful_chunks_;
    }
  } catch (const std::exception &) {
    // Exceptions may contain user content or filesystem paths: no raw log.
    failure_ = Qwen3ASRStreamingStatus::kDecodeError;
  }
  if (out->steps.size() > step_index) {
    auto &step = out->steps[step_index];
    step.service_seconds = std::chrono::duration<double>(
                               std::chrono::steady_clock::now() - started)
                               .count();
    if (failure_ == Qwen3ASRStreamingStatus::kDecodeError) {
      step.stop_reason = "decode_error";
    }
  }
  out->status = failure_;
  out->processed_samples = processed_;
  return failure_ == Qwen3ASRStreamingStatus::kOk;
}

Qwen3ASRStreamingOutcome Qwen3ASRStreamingState::AcceptWaveform(
    const float *samples, int32_t n) {
  using Status = Qwen3ASRStreamingStatus;
  if (failure_ != Status::kOk) return Outcome(failure_);
  if (finished_) return Outcome(Status::kInputFinished);
  if (n < 0 || (n > 0 && !samples)) return Outcome(Status::kInvalidArgument);
  if (static_cast<size_t>(n) > config_.max_utterance_samples - pcm_.size()) {
    return Outcome(Status::kAudioLimit);
  }
  for (int32_t i = 0; i < n; ++i) {
    if (!std::isfinite(samples[i]) || samples[i] < -1 || samples[i] > 1) {
      return Outcome(Status::kInvalidArgument);
    }
  }
  if (n == 0) return Outcome(Status::kOk);
  try {
    pcm_.insert(pcm_.end(), samples, samples + n);
  } catch (const std::exception &) {
    failure_ = Status::kDecodeError;
    return Outcome(failure_);
  }
  auto out = Outcome(Status::kOk);
  while (pcm_.size() - processed_ >=
         static_cast<size_t>(config_.chunk_size_samples)) {
    if (!DecodeAt(processed_ + config_.chunk_size_samples, &out)) break;
  }
  return out;
}

Qwen3ASRStreamingOutcome Qwen3ASRStreamingState::Finish() {
  using Status = Qwen3ASRStreamingStatus;
  auto out = Outcome(failure_);
  if (failure_ != Status::kOk || finished_) return out;
  if (pcm_.size() > static_cast<size_t>(processed_)) {
    if (!DecodeAt(static_cast<int32_t>(pcm_.size()), &out)) return out;
    result_.is_final = true;
    out.updates.back().is_final = true;
  } else {
    try {
      auto final = result_;
      final.is_final = true;
      ++final.revision;
      out.updates.push_back(final);
      result_ = std::move(final);
    } catch (const std::exception &) {
      failure_ = Status::kDecodeError;
      out.status = failure_;
      return out;
    }
  }
  finished_ = true;
  return out;
}

void Qwen3ASRStreamingState::Reset() {
  pcm_.clear();
  raw_ids_.clear();
  processed_ = 0;
  successful_chunks_ = 0;
  finished_ = false;
  failure_ = Qwen3ASRStreamingStatus::kOk;
  result_ = {};
}

Qwen3ASRSimulatedStreamingRecognizer::Qwen3ASRSimulatedStreamingRecognizer(
    const OfflineRecognizerConfig &model_config,
    const Qwen3ASRSimulatedStreamingConfig &config)
    : config_(config) {
  ValidateConfig(config_);
  const auto &m = model_config.model_config;
  const auto &q = m.qwen3_asr;
  ValidateHotwords(q.hotwords);
  if (q.conv_frontend.empty() || q.encoder.empty() || q.decoder.empty() ||
      q.tokenizer.empty() || (m.provider != "cpu" && !m.provider.empty()) ||
      (!m.model_type.empty() && m.model_type != "qwen3_asr") ||
      model_config.decoding_method != "greedy_search" ||
      !std::isfinite(q.temperature) || q.temperature < 0 ||
      q.temperature > 1e-6f || !std::isfinite(q.top_p) || q.top_p <= 0 ||
      q.top_p > 1 || q.max_new_tokens <= 0 || q.max_total_len <= 0 ||
      m.num_threads <= 0) {
    throw std::invalid_argument(
        "Qwen3-ASR streaming requires CPU greedy decoding");
  }
  engine_ = std::make_shared<OfflineRecognizerQwen3ASRImpl>(model_config);
}

std::unique_ptr<Qwen3ASRSimulatedStreamingStream>
Qwen3ASRSimulatedStreamingRecognizer::CreateStream(
    const Qwen3ASRSimulatedStreamingOptions &options) const {
  ValidateOptions(options);
  auto engine = engine_;
  auto decode = [engine, options](const std::vector<float> &pcm,
                                  const Qwen3ASRContinuationRequest &request) {
    auto snapshot = engine->CreateStream();
    snapshot->SetOption("language", options.language);
    if (options.hotwords) snapshot->SetOption("hotwords", *options.hotwords);
    snapshot->AcceptWaveform(16000, pcm.data(),
                             static_cast<int32_t>(pcm.size()));
    return engine->DecodeWithPrefix(snapshot.get(), request);
  };
  // Aliasing shared_ptr preserves the entire engine even after its owner dies.
  auto tokenizer =
      std::shared_ptr<QwenAsrTokenizer>(engine, &engine->Tokenizer());
  auto state =
      std::make_unique<Qwen3ASRStreamingState>(config_, tokenizer, decode);
  return std::unique_ptr<Qwen3ASRSimulatedStreamingStream>(
      new Qwen3ASRSimulatedStreamingStream(std::move(state)));
}

Qwen3ASRSimulatedStreamingStream::Qwen3ASRSimulatedStreamingStream(
    std::unique_ptr<Qwen3ASRStreamingState> state)
    : state_(std::move(state)) {}
Qwen3ASRSimulatedStreamingStream::~Qwen3ASRSimulatedStreamingStream() = default;
Qwen3ASRStreamingOutcome Qwen3ASRSimulatedStreamingStream::AcceptWaveform(
    const float *samples, int32_t n) {
  return state_->AcceptWaveform(samples, n);
}
Qwen3ASRStreamingOutcome Qwen3ASRSimulatedStreamingStream::Finish() {
  return state_->Finish();
}
void Qwen3ASRSimulatedStreamingStream::Reset() { state_->Reset(); }
Qwen3ASRStreamingUpdate Qwen3ASRSimulatedStreamingStream::GetResult() const {
  return state_->GetResult();
}
}  // namespace sherpa_onnx
