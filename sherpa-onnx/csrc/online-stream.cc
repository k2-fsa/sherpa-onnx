// sherpa-onnx/csrc/online-stream.cc
//
// Copyright (c)  2023  Xiaomi Corporation
#include "sherpa-onnx/csrc/online-stream.h"

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/features.h"
#include "sherpa-onnx/csrc/text-utils.h"
#include "sherpa-onnx/csrc/transducer-keyword-decoder.h"

namespace sherpa_onnx {

class OnlineStream::Impl {
 public:
  explicit Impl(const FeatureExtractorConfig &config,
                ContextGraphPtr context_graph)
      : feat_extractor_(config), context_graph_(std::move(context_graph)) {}

  void AcceptWaveform(int32_t sampling_rate, const float *waveform, int32_t n) {
    std::lock_guard<std::mutex> lock(mutex_);
    feat_extractor_.AcceptWaveform(sampling_rate, waveform, n);
  }

  void InputFinished() const {
    std::lock_guard<std::mutex> lock(mutex_);
    feat_extractor_.InputFinished();
  }

  int32_t NumFramesReady() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return feat_extractor_.NumFramesReady() - start_frame_index_;
  }

  bool IsLastFrame(int32_t frame) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return feat_extractor_.IsLastFrame(frame);
  }

  std::vector<float> GetFrames(int32_t frame_index, int32_t n) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return feat_extractor_.GetFrames(frame_index + start_frame_index_, n);
  }

  void Reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    // we don't reset the feature extractor
    start_frame_index_ += num_processed_frames_;
    num_processed_frames_ = 0;
  }

  int32_t &GetNumProcessedFrames() {
    std::lock_guard<std::mutex> lock(mutex_);
    return num_processed_frames_;
  }

  int32_t GetNumFramesSinceStart() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return start_frame_index_;
  }

  int32_t &GetCurrentSegment() {
    std::lock_guard<std::mutex> lock(mutex_);
    return segment_;
  }

  void SetResult(const OnlineTransducerDecoderResult &r) { result_ = r; }

  OnlineTransducerDecoderResult &GetResult() { return result_; }

  void SetKeywordResult(const TransducerKeywordResult &r) {
    keyword_result_ = r;
  }
  TransducerKeywordResult &GetKeywordResult(bool remove_duplicates) {
    if (remove_duplicates) {
      if (!prev_keyword_result_.timestamps.empty() &&
          !keyword_result_.timestamps.empty() &&
          keyword_result_.timestamps[0] <=
              prev_keyword_result_.timestamps.back()) {
        return empty_keyword_result_;
      } else {
        prev_keyword_result_ = keyword_result_;
      }
      return keyword_result_;
    } else {
      return keyword_result_;
    }
  }

  OnlineCtcDecoderResult &GetCtcResult() { return ctc_result_; }

  void SetCtcResult(const OnlineCtcDecoderResult &r) { ctc_result_ = r; }

  void SetParaformerResult(const OnlineParaformerDecoderResult &r) {
    paraformer_result_ = r;
  }

  OnlineParaformerDecoderResult &GetParaformerResult() {
    return paraformer_result_;
  }

  int32_t FeatureDim() const { return feat_extractor_.FeatureDim(); }

  void SetStates(std::vector<Ort::Value> states) {
    states_ = std::move(states);
  }

  std::vector<Ort::Value> &GetStates() { return states_; }

  void SetNeMoDecoderStates(std::vector<Ort::Value> decoder_states) {
    decoder_states_ = std::move(decoder_states);
  }

  std::vector<Ort::Value> &GetNeMoDecoderStates() { return decoder_states_; }

  const ContextGraphPtr &GetContextGraph() const { return context_graph_; }

  std::vector<float> &GetParaformerFeatCache() {
    return paraformer_feat_cache_;
  }

  std::vector<float> &GetParaformerEncoderOutCache() {
    return paraformer_encoder_out_cache_;
  }

  std::vector<float> &GetParaformerAlphaCache() {
    return paraformer_alpha_cache_;
  }

  void SetOption(const std::string &key, const std::string &value) {
    options_[key] = value;
  }

  bool HasOption(const std::string &key) const {
    return options_.count(key) != 0;
  }

  const std::string &GetOption(const std::string &key) const {
    auto it = options_.find(key);
    if (it != options_.end()) {
      return it->second;
    }
    static const std::string kEmpty;
    return kEmpty;
  }

  int32_t GetOptionInt(const std::string &key, int32_t default_value) const {
    auto it = options_.find(key);
    if (it != options_.end()) {
      return ToIntOrDefault(it->second, default_value);
    }
    return default_value;
  }

  float GetOptionFloat(const std::string &key, float default_value) const {
    auto it = options_.find(key);
    if (it != options_.end()) {
      return ToFloatOrDefault(it->second, default_value);
    }
    return default_value;
  }

  void SetFasterDecoder(std::unique_ptr<kaldi_decoder::FasterDecoder> decoder) {
    faster_decoder_ = std::move(decoder);
  }

  kaldi_decoder::FasterDecoder *GetFasterDecoder() const {
    return faster_decoder_.get();
  }

  int32_t &GetFasterDecoderProcessedFrames() {
    return faster_decoder_processed_frames_;
  }

 private:
  FeatureExtractor feat_extractor_;
  mutable std::mutex mutex_;
  /// For contextual-biasing
  ContextGraphPtr context_graph_;
  int32_t num_processed_frames_ = 0;  // before subsampling
  int32_t start_frame_index_ = 0;     // never reset
  int32_t segment_ = 0;
  OnlineTransducerDecoderResult result_;
  TransducerKeywordResult prev_keyword_result_;
  TransducerKeywordResult keyword_result_;
  TransducerKeywordResult empty_keyword_result_;
  OnlineCtcDecoderResult ctc_result_;
  std::vector<Ort::Value> states_;  // states for transducer or ctc models
  std::vector<Ort::Value> decoder_states_;  // states for nemo transducer models
  std::vector<float> paraformer_feat_cache_;
  std::vector<float> paraformer_encoder_out_cache_;
  std::vector<float> paraformer_alpha_cache_;
  OnlineParaformerDecoderResult paraformer_result_;
  std::unordered_map<std::string, std::string> options_;
  std::unique_ptr<kaldi_decoder::FasterDecoder> faster_decoder_;
  int32_t faster_decoder_processed_frames_ = 0;
};

OnlineStream::OnlineStream(const FeatureExtractorConfig &config /*= {}*/,
                           ContextGraphPtr context_graph /*= nullptr */)
    : impl_(std::make_unique<Impl>(config, std::move(context_graph))) {}

OnlineStream::~OnlineStream() = default;

void OnlineStream::AcceptWaveform(int32_t sampling_rate, const float *waveform,
                                  int32_t n) const {
  impl_->AcceptWaveform(sampling_rate, waveform, n);
}

void OnlineStream::InputFinished() const { impl_->InputFinished(); }

int32_t OnlineStream::NumFramesReady() const { return impl_->NumFramesReady(); }

bool OnlineStream::IsLastFrame(int32_t frame) const {
  return impl_->IsLastFrame(frame);
}

std::vector<float> OnlineStream::GetFrames(int32_t frame_index,
                                           int32_t n) const {
  return impl_->GetFrames(frame_index, n);
}

void OnlineStream::Reset() { impl_->Reset(); }

int32_t OnlineStream::FeatureDim() const { return impl_->FeatureDim(); }

int32_t &OnlineStream::GetNumProcessedFrames() {
  return impl_->GetNumProcessedFrames();
}

int32_t OnlineStream::GetNumFramesSinceStart() const {
  return impl_->GetNumFramesSinceStart();
}

int32_t &OnlineStream::GetCurrentSegment() {
  return impl_->GetCurrentSegment();
}

void OnlineStream::SetResult(const OnlineTransducerDecoderResult &r) {
  impl_->SetResult(r);
}

OnlineTransducerDecoderResult &OnlineStream::GetResult() {
  return impl_->GetResult();
}

void OnlineStream::SetKeywordResult(const TransducerKeywordResult &r) {
  impl_->SetKeywordResult(r);
}

TransducerKeywordResult &OnlineStream::GetKeywordResult(
    bool remove_duplicates /*=false*/) {
  return impl_->GetKeywordResult(remove_duplicates);
}

OnlineCtcDecoderResult &OnlineStream::GetCtcResult() {
  return impl_->GetCtcResult();
}

void OnlineStream::SetCtcResult(const OnlineCtcDecoderResult &r) {
  impl_->SetCtcResult(r);
}

void OnlineStream::SetParaformerResult(const OnlineParaformerDecoderResult &r) {
  impl_->SetParaformerResult(r);
}

OnlineParaformerDecoderResult &OnlineStream::GetParaformerResult() {
  return impl_->GetParaformerResult();
}

void OnlineStream::SetStates(std::vector<Ort::Value> states) {
  impl_->SetStates(std::move(states));
}

std::vector<Ort::Value> &OnlineStream::GetStates() {
  return impl_->GetStates();
}

void OnlineStream::SetNeMoDecoderStates(
    std::vector<Ort::Value> decoder_states) {
  return impl_->SetNeMoDecoderStates(std::move(decoder_states));
}

std::vector<Ort::Value> &OnlineStream::GetNeMoDecoderStates() {
  return impl_->GetNeMoDecoderStates();
}

const ContextGraphPtr &OnlineStream::GetContextGraph() const {
  return impl_->GetContextGraph();
}

void OnlineStream::SetFasterDecoder(
    std::unique_ptr<kaldi_decoder::FasterDecoder> decoder) {
  impl_->SetFasterDecoder(std::move(decoder));
}

kaldi_decoder::FasterDecoder *OnlineStream::GetFasterDecoder() const {
  return impl_->GetFasterDecoder();
}

int32_t &OnlineStream::GetFasterDecoderProcessedFrames() {
  return impl_->GetFasterDecoderProcessedFrames();
}

std::vector<float> &OnlineStream::GetParaformerFeatCache() {
  return impl_->GetParaformerFeatCache();
}

std::vector<float> &OnlineStream::GetParaformerEncoderOutCache() {
  return impl_->GetParaformerEncoderOutCache();
}

std::vector<float> &OnlineStream::GetParaformerAlphaCache() {
  return impl_->GetParaformerAlphaCache();
}

void OnlineStream::SetOption(const std::string &key,
                             const std::string &value) {
  impl_->SetOption(key, value);
}

bool OnlineStream::HasOption(const std::string &key) const {
  return impl_->HasOption(key);
}

const std::string &OnlineStream::GetOption(const std::string &key) const {
  return impl_->GetOption(key);
}

int32_t OnlineStream::GetOptionInt(const std::string &key,
                                   int32_t default_value) const {
  return impl_->GetOptionInt(key, default_value);
}

float OnlineStream::GetOptionFloat(const std::string &key,
                                   float default_value) const {
  return impl_->GetOptionFloat(key, default_value);
}

}  // namespace sherpa_onnx
