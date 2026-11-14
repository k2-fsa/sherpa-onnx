// sherpa-onnx/csrc/online-recognizer.cc
//
// Copyright (c)  2023  Xiaomi Corporation
// Copyright (c)  2023  Pingfeng Luo

#include "sherpa-onnx/csrc/online-recognizer.h"

#include <algorithm>
#include <cassert>
#include <iomanip>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#if __OHOS__
#include "rawfile/raw_file_manager.h"
#endif

#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/online-recognizer-impl.h"
#include "sherpa-onnx/csrc/online-vad-model.h"
#include "sherpa-onnx/csrc/text-utils.h"

namespace sherpa_onnx {

namespace {

/// Helper for `OnlineRecognizerResult::AsJsonString()`
template <typename T>
std::string VecToString(const std::vector<T> &vec, int32_t precision = 6) {
  std::ostringstream oss;
  if (precision != 0) {
    oss << std::fixed << std::setprecision(precision);
  }
  oss << "[";
  std::string sep = "";
  for (const auto &item : vec) {
    oss << sep << item;
    sep = ", ";
  }
  oss << "]";
  return oss.str();
}

/// Helper for `OnlineRecognizerResult::AsJsonString()`
template <>  // explicit specialization for T = std::string
std::string VecToString<std::string>(const std::vector<std::string> &vec,
                                     int32_t) {  // ignore 2nd arg
  std::ostringstream oss;
  oss << "[";
  std::string sep = "";
  for (const auto &item : vec) {
    oss << sep << QuoteJsonString(item);
    sep = ", ";
  }
  oss << "]";
  return oss.str();
}

}  // namespace

std::string OnlineRecognizerResult::AsJsonString() const {
  std::ostringstream os;
  os << "{ ";
  os << "\"text\": " << QuoteJsonString(text) << ", ";
  os << "\"tokens\": " << VecToString(tokens) << ", ";
  os << "\"timestamps\": " << VecToString(timestamps, 2) << ", ";
  os << "\"ys_probs\": " << VecToString(ys_probs, 6) << ", ";
  os << "\"lm_probs\": " << VecToString(lm_probs, 6) << ", ";
  os << "\"context_scores\": " << VecToString(context_scores, 6) << ", ";
  os << "\"segment\": " << segment << ", ";
  os << "\"words\": " << VecToString(words, 0) << ", ";
  os << "\"num_trailing_blanks\": " << num_trailing_blanks << ", ";
  os << "\"endpoint_rule\": " << endpoint_rule << ", ";
  os << "\"endpoint_delay\": " << std::fixed << std::setprecision(2)
     << endpoint_delay << ", ";
  os << "\"speech_end_time\": " << std::fixed << std::setprecision(2)
     << speech_end_time << ", ";
  os << "\"start_time\": " << std::fixed << std::setprecision(2) << start_time
     << ", ";
  os << "\"is_final\": " << (is_final ? "true" : "false") << ", ";
  os << "\"is_eof\": " << (is_eof ? "true" : "false");
  os << "}";
  return os.str();
}

void OnlineRecognizerConfig::Register(ParseOptions *po) {
  feat_config.Register(po);
  model_config.Register(po);
  endpoint_config.Register(po);
  lm_config.Register(po);
  ctc_fst_decoder_config.Register(po);
  online_vad_config.Register(po);
  hr.Register(po);

  po->Register("enable-endpoint", &enable_endpoint,
               "True to enable endpoint detection. False to disable it.");
  po->Register("max-active-paths", &max_active_paths,
               "beam size used in modified beam search.");
  po->Register("blank-penalty", &blank_penalty,
               "The penalty applied on blank symbol during decoding. "
               "Note: It is a positive value. "
               "Increasing value will lead to lower deletion at the cost"
               "of higher insertions. "
               "Currently only applicable for transducer models.");
  po->Register("hotwords-score", &hotwords_score,
               "The bonus score for each token in context word/phrase. "
               "Used only when decoding_method is modified_beam_search");
  po->Register(
      "hotwords-file", &hotwords_file,
      "The file containing hotwords, one words/phrases per line, For example: "
      "HELLO WORLD"
      "你好世界");
  po->Register("decoding-method", &decoding_method,
               "decoding method,"
               "now support greedy_search and modified_beam_search.");
  po->Register("temperature-scale", &temperature_scale,
               "Temperature scale for confidence computation in decoding.");
  po->Register(
      "rule-fsts", &rule_fsts,
      "If not empty, it specifies fsts for inverse text normalization. "
      "If there are multiple fsts, they are separated by a comma.");

  po->Register(
      "rule-fars", &rule_fars,
      "If not empty, it specifies fst archives for inverse text normalization. "
      "If there are multiple archives, they are separated by a comma.");

  po->Register("reset-encoder", &reset_encoder,
               "True to reset encoder_state on an endpoint after empty segment."
               "Done in `Reset()` method, after an endpoint was detected.");
}

bool OnlineRecognizerConfig::Validate() const {
  if (decoding_method == "modified_beam_search") {
    if (max_active_paths <= 0) {
      SHERPA_ONNX_LOGE("max_active_paths must be > 0. Given: %d",
                       max_active_paths);
      return false;
    }
  }

  if (decoding_method == "modified_beam_search" && !lm_config.model.empty()) {
    if (!lm_config.Validate()) {
      return false;
    }
  }

  if (!hotwords_file.empty() && decoding_method != "modified_beam_search") {
    SHERPA_ONNX_LOGE(
        "Please use --decoding-method=modified_beam_search if you"
        " provide --hotwords-file. Given --decoding-method=%s",
        decoding_method.c_str());
    return false;
  }

  if (!ctc_fst_decoder_config.graph.empty() &&
      !ctc_fst_decoder_config.Validate()) {
    SHERPA_ONNX_LOGE("Errors in ctc_fst_decoder_config");
    return false;
  }

  if (!hotwords_file.empty() && !FileExists(hotwords_file)) {
    SHERPA_ONNX_LOGE("--hotwords-file: '%s' does not exist",
                     hotwords_file.c_str());
    return false;
  }

  if (!rule_fsts.empty()) {
    std::vector<std::string> files;
    SplitStringToVector(rule_fsts, ",", false, &files);
    for (const auto &f : files) {
      if (!FileExists(f)) {
        SHERPA_ONNX_LOGE("Rule fst '%s' does not exist. ", f.c_str());
        return false;
      }
    }
  }

  if (!rule_fars.empty()) {
    std::vector<std::string> files;
    SplitStringToVector(rule_fars, ",", false, &files);
    for (const auto &f : files) {
      if (!FileExists(f)) {
        SHERPA_ONNX_LOGE("Rule far '%s' does not exist. ", f.c_str());
        return false;
      }
    }
  }

  if (!hr.lexicon.empty() && !hr.rule_fsts.empty() && !hr.Validate()) {
    return false;
  }

  if (online_vad_config.IsEnabled() && !online_vad_config.Validate()) {
    SHERPA_ONNX_LOGE("Errors in online_vad_config");
    return false;
  }

  return model_config.Validate();
}

std::string OnlineRecognizerConfig::ToString() const {
  std::ostringstream os;

  os << "OnlineRecognizerConfig(";
  os << "feat_config=" << feat_config.ToString() << ", ";
  os << "model_config=" << model_config.ToString() << ", ";
  os << "lm_config=" << lm_config.ToString() << ", ";
  os << "endpoint_config=" << endpoint_config.ToString() << ", ";
  os << "ctc_fst_decoder_config=" << ctc_fst_decoder_config.ToString() << ", ";
  os << "online_vad_config=" << online_vad_config.ToString() << ", ";
  os << "enable_endpoint=" << (enable_endpoint ? "True" : "False") << ", ";
  os << "max_active_paths=" << max_active_paths << ", ";
  os << "hotwords_score=" << hotwords_score << ", ";
  os << "hotwords_file=\"" << hotwords_file << "\", ";
  os << "decoding_method=\"" << decoding_method << "\", ";
  os << "blank_penalty=" << blank_penalty << ", ";
  os << "temperature_scale=" << temperature_scale << ", ";
  os << "rule_fsts=\"" << rule_fsts << "\", ";
  os << "rule_fars=\"" << rule_fars << "\", ";
  os << "reset_encoder=" << (reset_encoder ? "True" : "False") << ", ";
  os << "hr=" << hr.ToString() << ")";

  return os.str();
}

OnlineRecognizer::OnlineRecognizer(const OnlineRecognizerConfig &config)
    : impl_(OnlineRecognizerImpl::Create(config)),
      online_vad_config_(config.online_vad_config),
      asr_frame_shift_ms_(config.feat_config.frame_shift_ms),
      vad_model_(OnlineVadModel::Create(config.online_vad_config)) {}

template <typename Manager>
OnlineRecognizer::OnlineRecognizer(Manager *mgr,
                                   const OnlineRecognizerConfig &config)
    : impl_(OnlineRecognizerImpl::Create(mgr, config)),
      online_vad_config_(config.online_vad_config),
      asr_frame_shift_ms_(config.feat_config.frame_shift_ms),
      vad_model_(OnlineVadModel::Create(config.online_vad_config)) {}

OnlineRecognizer::~OnlineRecognizer() = default;

std::unique_ptr<OnlineStream> OnlineRecognizer::CreateStream() const {
  auto s = impl_->CreateStream();
  InitStreamVad(s.get());
  return s;
}

std::unique_ptr<OnlineStream> OnlineRecognizer::CreateStream(
    const std::string &hotwords) const {
  auto s = impl_->CreateStream(hotwords);
  InitStreamVad(s.get());
  return s;
}

void OnlineRecognizer::InitStreamVad(OnlineStream *s) const {
  if (!vad_model_) return;
  // Decoders without chunk geometry cannot flush.
  const bool flush_supported = impl_->GetChunkSizeForVad() > 0;
  s->InitVad(vad_model_->SampleRate(), vad_model_->WindowShift(),
             vad_model_->WindowSize(), asr_frame_shift_ms_, flush_supported);
}

void OnlineRecognizer::VadProcess(OnlineStream **ss, int32_t n) const {
  if (!vad_model_) return;
  vad_model_->RunVad(ss, n);
}

bool OnlineRecognizer::IsDecodable(OnlineStream *s) const {
  if (s->HasVad() && s->IsVadFlushPending()) return true;
  return impl_->IsReady(s);
}

bool OnlineRecognizer::VadHasWindow(OnlineStream *s) const {
  return vad_model_ && s->VadPendingSize() >= vad_model_->WindowSize();
}

bool OnlineRecognizer::IsReady(OnlineStream *s) const {
  if (s->HasVad() && s->IsVadFlushPending()) {
    // VAD-confirmed end of speech: decode what is left now (DecodeStreams()
    // pads the last chunk with silence), without the VAD coverage gate.
    int32_t chunk_size = impl_->GetChunkSizeForVad();
    int32_t remaining = s->GetVadEndpointFrame() - s->GetNumProcessedFrames();
    if (remaining <= chunk_size) return true;
    return s->GetNumProcessedFrames() + chunk_size < s->NumFramesReady();
  }

  bool ready = impl_->IsReady(s);
  if (!ready || !s->HasVad() || s->IsInputFinished()) {
    return ready;
  }

  MaybeSkipVadSilence(s);

  // MaybeSkipVadSilence() may have just advanced num_processed_frames_,
  // but `ready` above was computed *before* that, against the old
  // position. Re-validate against real fbank-frame availability at the
  // new position before trusting it: the VAD coverage check below is an
  // *additional* requirement on top of real readiness, never a
  // substitute for it. Skipping this re-check was the actual root cause
  // of a crash found under sustained load (features.cc:GetFrames
  // tripping "frame_index + n <= NumFramesReady()", confirmed via an
  // AddressSanitizer-instrumented build) -- the per-step clamp inside
  // MaybeSkipVadSilence only protects each chunk_shift-sized step of the
  // skip loop itself, not the chunk_size-sized window this function
  // then promises the caller is safe to decode.
  ready = impl_->IsReady(s);
  if (!ready) {
    return false;
  }

  int32_t chunk_size = impl_->GetChunkSizeForVad();
  if (chunk_size <= 0) {
    // This decoder impl doesn't support VAD-coverage gating (e.g.
    // Paraformer) -- fall back to the blank-frame-count readiness check
    // only.
    return ready;
  }

  int64_t target = static_cast<int64_t>(s->GetNumFramesSinceStart()) +
                   s->GetNumProcessedFrames() + chunk_size;
  int64_t covered = s->GetVadFramesCoveredAbsolute();
  return covered >= target;
}

void OnlineRecognizer::MaybeSkipVadSilence(OnlineStream *s) const {
  if (!s->HasVad() || s->IsInputFinished() || !online_vad_config_.batch_skip) {
    return;
  }

  // Frames up to an armed endpoint belong to the flush.
  if (s->IsVadEndpointArmed()) {
    return;
  }

  int32_t chunk_size = impl_->GetChunkSizeForVad();
  int32_t chunk_shift = impl_->GetChunkShiftForVad();
  if (chunk_size <= 0 || chunk_shift <= 0) {
    return;
  }

  int64_t silence_start = 0;
  if (!s->GetVadConfirmedSilenceStartAbsolute(&silence_start)) {
    return;
  }
  int64_t vad_covered = s->GetVadFramesCoveredAbsolute();

  while (true) {
    int32_t local_processed = s->GetNumProcessedFrames();
    // Safety clamp: never advance past what the feature extractor has
    // actually buffered. The VAD's own coverage estimate
    // (vad_covered, derived from window-count * window-shift-seconds)
    // is necessarily approximate -- trusting it alone to gate how far
    // num_processed_frames_ moves risks running DecodeStreams/GetFrames
    // past real data and crashing (found via load testing: a long-
    // running load test eventually hit `features.cc:GetFrames` tripping
    // its "frame_index + n <= NumFramesReady()" assertion).
    if (local_processed + chunk_shift > s->NumFramesReady()) {
      break;
    }

    int64_t window_start =
        static_cast<int64_t>(s->GetNumFramesSinceStart()) + local_processed;
    int64_t window_end = window_start + chunk_size;
    if (window_start < silence_start || window_end > vad_covered) {
      break;
    }
    s->GetNumProcessedFrames() += chunk_shift;
  }
}

void OnlineRecognizer::WarmpUpRecognizer(int32_t warmup, int32_t mbs) const {
  if (warmup > 0) {
    impl_->WarmpUpRecognizer(warmup, mbs);
    if (vad_model_) {
      vad_model_->WarmUp(warmup);
    }
  }
}

void OnlineRecognizer::DecodeStreams(OnlineStream **ss, int32_t n) const {
  impl_->DecodeStreams(ss, n);
}

OnlineRecognizerResult OnlineRecognizer::GetResult(OnlineStream *s) const {
  auto r = impl_->GetResult(s);
  r.text = RemoveLeadingSpaces(r.text);
  RemoveLeadingPunct(&r);
  r.endpoint_rule = s->GetEndpointRule();
  r.endpoint_delay = s->GetEndpointDelay();
  r.speech_end_time = s->GetSpeechEndTime();
  return r;
}

void RemoveLeadingPunct(OnlineRecognizerResult *r) {
  std::string text = RemoveLeadingPunct(r->text);
  if (text.size() == r->text.size()) {
    return;
  }

  std::string punct = r->text.substr(0, r->text.size() - text.size());
  r->text = std::move(text);

  // The first token with visible content must start with the removed
  // punctuation. Otherwise, e.g., for byte fallback or byte-level BPE
  // tokens, leave tokens alone.
  auto &tokens = r->tokens;
  for (size_t i = 0; i != tokens.size(); ++i) {
    size_t start = tokens[i].find_first_not_of(' ');
    if (start == std::string::npos) {
      continue;
    }

    if (tokens[i].compare(start, punct.size(), punct) != 0) {
      return;
    }

    if (start + punct.size() < tokens[i].size()) {
      tokens[i].erase(start, punct.size());
      return;
    }

    int32_t n = static_cast<int32_t>(tokens.size());
    auto erase = [i, n](auto *v) {
      if (static_cast<int32_t>(v->size()) == n) {
        v->erase(v->begin() + i);
      }
    };

    erase(&r->timestamps);
    erase(&r->ys_probs);
    erase(&r->lm_probs);
    erase(&r->context_scores);
    tokens.erase(tokens.begin() + i);
    return;
  }
}

bool OnlineRecognizer::IsEndpoint(OnlineStream *s) const {
  if (s->HasVad() && s->IsVadEndpoint()) {
    s->GetEndpointRule() = 4;

    int64_t covered = s->GetVadFramesCoveredAbsolute();
    int64_t last_speech = 0;
    bool had_speech = s->GetVadConfirmedSilenceStartAbsolute(&last_speech);
    float frame_shift_s = asr_frame_shift_ms_ / 1000.0f;
    if (had_speech) {
      s->GetSpeechEndTime() = last_speech * frame_shift_s;
      s->GetEndpointDelay() = (covered - last_speech) * frame_shift_s;
    }
    return true;
  }
  if (impl_->IsEndpoint(s)) {
    s->GetEndpointRule() = 1;
    return true;
  }
  return false;
}

void OnlineRecognizer::Reset(OnlineStream *s) const {
  impl_->Reset(s);
  if (s->HasVad()) {
    s->ResetVad();
  }
  s->GetEndpointRule() = 0;
  s->GetEndpointDelay() = 0;
  s->GetSpeechEndTime() = 0;
}

#if __ANDROID_API__ >= 9
template OnlineRecognizer::OnlineRecognizer(
    AAssetManager *mgr, const OnlineRecognizerConfig &config);
#endif

#if __OHOS__
template OnlineRecognizer::OnlineRecognizer(
    NativeResourceManager *mgr, const OnlineRecognizerConfig &config);
#endif

}  // namespace sherpa_onnx
