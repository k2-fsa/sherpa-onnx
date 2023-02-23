// sherpa-onnx/csrc/online-recognizer.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/online-recognizer.h"

#include <assert.h>

#include <algorithm>
#include <memory>
#include <sstream>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/online-transducer-decoder.h"
#include "sherpa-onnx/csrc/online-transducer-greedy-search-decoder.h"
#include "sherpa-onnx/csrc/online-transducer-model.h"
#include "sherpa-onnx/csrc/symbol-table.h"

namespace sherpa_onnx {

static OnlineRecognizerResult Convert(const OnlineTransducerDecoderResult &src,
                                      const SymbolTable &sym_table) {
  std::string text;
  for (auto t : src.tokens) {
    text += sym_table[t];
  }

  OnlineRecognizerResult ans;
  ans.text = std::move(text);
  return ans;
}

std::string OnlineRecognizerConfig::ToString() const {
  std::ostringstream os;

  os << "OnlineRecognizerConfig(";
  os << "feat_config=" << feat_config.ToString() << ", ";
  os << "model_config=" << model_config.ToString() << ", ";
  os << "endpoint_config=" << endpoint_config.ToString() << ", ";
  os << "enable_endpoint=" << (enable_endpoint ? "True" : "False") << ")";

  return os.str();
}

class OnlineRecognizer::Impl {
 public:
  explicit Impl(const OnlineRecognizerConfig &config)
      : config_(config),
        model_(OnlineTransducerModel::Create(config.model_config)),
        sym_(config.model_config.tokens),
        endpoint_(config_.endpoint_config) {
    decoder_ =
        std::make_unique<OnlineTransducerGreedySearchDecoder>(model_.get());
  }

#if __ANDROID_API__ >= 9
  explicit Impl(AAssetManager *mgr, const OnlineRecognizerConfig &config)
      : config_(config),
        model_(OnlineTransducerModel::Create(mgr, config.model_config)),
        sym_(mgr, config.model_config.tokens),
        endpoint_(config_.endpoint_config) {
    decoder_ =
        std::make_unique<OnlineTransducerGreedySearchDecoder>(model_.get());
  }
#endif

  std::unique_ptr<OnlineStream> CreateStream() const {
    auto stream = std::make_unique<OnlineStream>(config_.feat_config);
    stream->SetResult(decoder_->GetEmptyResult());
    stream->SetStates(model_->GetEncoderInitStates());
    return stream;
  }

  bool IsReady(OnlineStream *s) const {
    return s->GetNumProcessedFrames() + model_->ChunkSize() <
           s->NumFramesReady();
  }

  void DecodeStreams(OnlineStream **ss, int32_t n) const {
    int32_t chunk_size = model_->ChunkSize();
    int32_t chunk_shift = model_->ChunkShift();

    int32_t feature_dim = ss[0]->FeatureDim();

    std::vector<OnlineTransducerDecoderResult> results(n);
    std::vector<float> features_vec(n * chunk_size * feature_dim);
    std::vector<std::vector<Ort::Value>> states_vec(n);

    for (int32_t i = 0; i != n; ++i) {
      std::vector<float> features =
          ss[i]->GetFrames(ss[i]->GetNumProcessedFrames(), chunk_size);

      ss[i]->GetNumProcessedFrames() += chunk_shift;

      std::copy(features.begin(), features.end(),
                features_vec.data() + i * chunk_size * feature_dim);

      results[i] = std::move(ss[i]->GetResult());
      states_vec[i] = std::move(ss[i]->GetStates());
    }

    auto memory_info =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

    std::array<int64_t, 3> x_shape{n, chunk_size, feature_dim};

    Ort::Value x = Ort::Value::CreateTensor(memory_info, features_vec.data(),
                                            features_vec.size(), x_shape.data(),
                                            x_shape.size());

    auto states = model_->StackStates(states_vec);

    auto pair = model_->RunEncoder(std::move(x), std::move(states));

    decoder_->Decode(std::move(pair.first), &results);

    std::vector<std::vector<Ort::Value>> next_states =
        model_->UnStackStates(pair.second);

    for (int32_t i = 0; i != n; ++i) {
      ss[i]->SetResult(results[i]);
      ss[i]->SetStates(std::move(next_states[i]));
    }
  }

  OnlineRecognizerResult GetResult(OnlineStream *s) const {
    OnlineTransducerDecoderResult decoder_result = s->GetResult();
    decoder_->StripLeadingBlanks(&decoder_result);

    return Convert(decoder_result, sym_);
  }

  bool IsEndpoint(OnlineStream *s) const {
    if (!config_.enable_endpoint) return false;
    int32_t num_processed_frames = s->GetNumProcessedFrames();

    // frame shift is 10 milliseconds
    float frame_shift_in_seconds = 0.01;

    // subsampling factor is 4
    int32_t trailing_silence_frames = s->GetResult().num_trailing_blanks * 4;

    return endpoint_.IsEndpoint(num_processed_frames, trailing_silence_frames,
                                frame_shift_in_seconds);
  }

  void Reset(OnlineStream *s) const {
    // reset result and neural network model state,
    // but keep the feature extractor state

    // reset result
    s->SetResult(decoder_->GetEmptyResult());

    // reset neural network model state
    s->SetStates(model_->GetEncoderInitStates());
  }

 private:
  OnlineRecognizerConfig config_;
  std::unique_ptr<OnlineTransducerModel> model_;
  std::unique_ptr<OnlineTransducerDecoder> decoder_;
  SymbolTable sym_;
  Endpoint endpoint_;
};

OnlineRecognizer::OnlineRecognizer(const OnlineRecognizerConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

#if __ANDROID_API__ >= 9
OnlineRecognizer::OnlineRecognizer(AAssetManager *mgr,
                                   const OnlineRecognizerConfig &config)
    : impl_(std::make_unique<Impl>(mgr, config)) {}
#endif

OnlineRecognizer::~OnlineRecognizer() = default;

std::unique_ptr<OnlineStream> OnlineRecognizer::CreateStream() const {
  return impl_->CreateStream();
}

bool OnlineRecognizer::IsReady(OnlineStream *s) const {
  return impl_->IsReady(s);
}

void OnlineRecognizer::DecodeStreams(OnlineStream **ss, int32_t n) const {
  impl_->DecodeStreams(ss, n);
}

OnlineRecognizerResult OnlineRecognizer::GetResult(OnlineStream *s) const {
  return impl_->GetResult(s);
}

bool OnlineRecognizer::IsEndpoint(OnlineStream *s) const {
  return impl_->IsEndpoint(s);
}

void OnlineRecognizer::Reset(OnlineStream *s) const { impl_->Reset(s); }

}  // namespace sherpa_onnx
