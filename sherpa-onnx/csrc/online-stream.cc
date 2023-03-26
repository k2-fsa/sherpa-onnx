// sherpa-onnx/csrc/online-stream.cc
//
// Copyright (c)  2023  Xiaomi Corporation
#include "sherpa-onnx/csrc/online-stream.h"

#include <memory>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/features.h"

namespace sherpa_onnx {

class OnlineStream::Impl {
 public:
  explicit Impl(const FeatureExtractorConfig &config)
      : feat_extractor_(config) {}

  void AcceptWaveform(int32_t sampling_rate, const float *waveform, int32_t n) {
    feat_extractor_.AcceptWaveform(sampling_rate, waveform, n);
  }

  void InputFinished() const { feat_extractor_.InputFinished(); }

  int32_t NumFramesReady() const {
    return feat_extractor_.NumFramesReady() - start_frame_index_;
  }

  bool IsLastFrame(int32_t frame) const {
    return feat_extractor_.IsLastFrame(frame);
  }

  std::vector<float> GetFrames(int32_t frame_index, int32_t n) const {
    return feat_extractor_.GetFrames(frame_index + start_frame_index_, n);
  }

  void Reset() {
    // we don't reset the feature extractor
    start_frame_index_ += num_processed_frames_;
    num_processed_frames_ = 0;
  }

  int32_t &GetNumProcessedFrames() { return num_processed_frames_; }

  void SetResult(const OnlineTransducerDecoderResult &r) { result_ = r; }

  OnlineTransducerDecoderResult &GetResult() { return result_; }

  int32_t FeatureDim() const { return feat_extractor_.FeatureDim(); }

  void SetStates(std::vector<Ort::Value> states) {
    states_ = std::move(states);
  }

  std::vector<Ort::Value> &GetStates() { return states_; }

 private:
  FeatureExtractor feat_extractor_;
  int32_t num_processed_frames_ = 0;  // before subsampling
  int32_t start_frame_index_ = 0;     // never reset
  OnlineTransducerDecoderResult result_;
  std::vector<Ort::Value> states_;
};

OnlineStream::OnlineStream(const FeatureExtractorConfig &config /*= {}*/)
    : impl_(std::make_unique<Impl>(config)) {}

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

void OnlineStream::SetResult(const OnlineTransducerDecoderResult &r) {
  impl_->SetResult(r);
}

OnlineTransducerDecoderResult &OnlineStream::GetResult() {
  return impl_->GetResult();
}

void OnlineStream::SetStates(std::vector<Ort::Value> states) {
  impl_->SetStates(std::move(states));
}

std::vector<Ort::Value> &OnlineStream::GetStates() {
  return impl_->GetStates();
}

}  // namespace sherpa_onnx
