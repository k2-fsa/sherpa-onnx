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
  os << "tokens=\"" << tokens << "\")";

  return os.str();
}

class OnlineRecognizer::Impl {
 public:
  explicit Impl(const OnlineRecognizerConfig &config)
      : config_(config),
        model_(OnlineTransducerModel::Create(config.model_config)),
        sym_(config.tokens) {
    decoder_ =
        std::make_unique<OnlineTransducerGreedySearchDecoder>(model_.get());
  }

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

  void DecodeStreams(OnlineStream **ss, int32_t n) {
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

  OnlineRecognizerResult GetResult(OnlineStream *s) {
    OnlineTransducerDecoderResult decoder_result = s->GetResult();
    decoder_->StripLeadingBlanks(&decoder_result);

    return Convert(decoder_result, sym_);
  }

 private:
  OnlineRecognizerConfig config_;
  std::unique_ptr<OnlineTransducerModel> model_;
  std::unique_ptr<OnlineTransducerDecoder> decoder_;
  SymbolTable sym_;
};

OnlineRecognizer::OnlineRecognizer(const OnlineRecognizerConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}
OnlineRecognizer::~OnlineRecognizer() = default;

std::unique_ptr<OnlineStream> OnlineRecognizer::CreateStream() const {
  return impl_->CreateStream();
}

bool OnlineRecognizer::IsReady(OnlineStream *s) const {
  return impl_->IsReady(s);
}

void OnlineRecognizer::DecodeStreams(OnlineStream **ss, int32_t n) {
  impl_->DecodeStreams(ss, n);
}

OnlineRecognizerResult OnlineRecognizer::GetResult(OnlineStream *s) {
  return impl_->GetResult(s);
}

}  // namespace sherpa_onnx
