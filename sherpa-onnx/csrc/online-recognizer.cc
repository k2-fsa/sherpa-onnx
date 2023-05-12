// sherpa-onnx/csrc/online-recognizer.cc
//
// Copyright (c)  2023  Xiaomi Corporation
// Copyright (c)  2023  Pingfeng Luo

#include "sherpa-onnx/csrc/online-recognizer.h"

#include <assert.h>

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <utility>
#include <vector>

#include "nlohmann/json.hpp"
#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/online-lm.h"
#include "sherpa-onnx/csrc/online-transducer-decoder.h"
#include "sherpa-onnx/csrc/online-transducer-greedy-search-decoder.h"
#include "sherpa-onnx/csrc/online-transducer-model.h"
#include "sherpa-onnx/csrc/online-transducer-modified-beam-search-decoder.h"
#include "sherpa-onnx/csrc/symbol-table.h"

namespace sherpa_onnx {

std::string OnlineRecognizerResult::AsJsonString() const {
  using json = nlohmann::json;
  json j;
  j["text"] = text;
  j["tokens"] = tokens;
  j["start_time"] = start_time;
#if 1
  // This branch chooses number of decimal points to keep in
  // the return json string
  std::ostringstream os;
  os << "[";
  std::string sep = "";
  for (auto t : timestamps) {
    os << sep << std::fixed << std::setprecision(2) << t;
    sep = ", ";
  }
  os << "]";
  j["timestamps"] = os.str();
#else
  j["timestamps"] = timestamps;
#endif

  j["segment"] = segment;
  j["is_final"] = is_final;

  return j.dump();
}

static OnlineRecognizerResult Convert(const OnlineTransducerDecoderResult &src,
                                      const SymbolTable &sym_table,
                                      int32_t frame_shift_ms,
                                      int32_t subsampling_factor) {
  OnlineRecognizerResult r;
  r.tokens.reserve(src.tokens.size());
  r.timestamps.reserve(src.tokens.size());

  for (auto i : src.tokens) {
    auto sym = sym_table[i];

    r.text.append(sym);
    r.tokens.push_back(std::move(sym));
  }

  float frame_shift_s = frame_shift_ms / 1000. * subsampling_factor;
  for (auto t : src.timestamps) {
    float time = frame_shift_s * t;
    r.timestamps.push_back(time);
  }

  return r;
}

void OnlineRecognizerConfig::Register(ParseOptions *po) {
  feat_config.Register(po);
  model_config.Register(po);
  endpoint_config.Register(po);
  lm_config.Register(po);

  po->Register("enable-endpoint", &enable_endpoint,
               "True to enable endpoint detection. False to disable it.");
  po->Register("max-active-paths", &max_active_paths,
               "beam size used in modified beam search.");
  po->Register("decoding-method", &decoding_method,
               "decoding method,"
               "now support greedy_search and modified_beam_search.");
}

bool OnlineRecognizerConfig::Validate() const {
  if (decoding_method == "modified_beam_search" && !lm_config.model.empty()) {
    if (max_active_paths <= 0) {
      SHERPA_ONNX_LOGE("max_active_paths is less than 0! Given: %d",
                       max_active_paths);
      return false;
    }
    if (!lm_config.Validate()) return false;
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
  os << "enable_endpoint=" << (enable_endpoint ? "True" : "False") << ", ";
  os << "max_active_paths=" << max_active_paths << ", ";
  os << "decoding_method=\"" << decoding_method << "\")";

  return os.str();
}

class OnlineRecognizer::Impl {
 public:
  explicit Impl(const OnlineRecognizerConfig &config)
      : config_(config),
        model_(OnlineTransducerModel::Create(config.model_config)),
        sym_(config.model_config.tokens),
        endpoint_(config_.endpoint_config) {
    if (config.decoding_method == "modified_beam_search") {
      if (!config_.lm_config.model.empty()) {
        lm_ = OnlineLM::Create(config.lm_config);
      }

      decoder_ = std::make_unique<OnlineTransducerModifiedBeamSearchDecoder>(
          model_.get(), lm_.get(), config_.max_active_paths,
          config_.lm_config.scale);
    } else if (config.decoding_method == "greedy_search") {
      decoder_ =
          std::make_unique<OnlineTransducerGreedySearchDecoder>(model_.get());
    } else {
      fprintf(stderr, "Unsupported decoding method: %s\n",
              config.decoding_method.c_str());
      exit(-1);
    }
  }

#if __ANDROID_API__ >= 9
  explicit Impl(AAssetManager *mgr, const OnlineRecognizerConfig &config)
      : config_(config),
        model_(OnlineTransducerModel::Create(mgr, config.model_config)),
        sym_(mgr, config.model_config.tokens),
        endpoint_(config_.endpoint_config) {
    if (config.decoding_method == "modified_beam_search") {
      decoder_ = std::make_unique<OnlineTransducerModifiedBeamSearchDecoder>(
          model_.get(), lm_.get(), config_.max_active_paths,
          config_.lm_config.scale);
    } else if (config.decoding_method == "greedy_search") {
      decoder_ =
          std::make_unique<OnlineTransducerGreedySearchDecoder>(model_.get());
    } else {
      fprintf(stderr, "Unsupported decoding method: %s\n",
              config.decoding_method.c_str());
      exit(-1);
    }
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
    std::vector<int64_t> all_processed_frames(n);

    for (int32_t i = 0; i != n; ++i) {
      const auto num_processed_frames = ss[i]->GetNumProcessedFrames();
      std::vector<float> features =
          ss[i]->GetFrames(num_processed_frames, chunk_size);

      // Question: should num_processed_frames include chunk_shift?
      ss[i]->GetNumProcessedFrames() += chunk_shift;

      std::copy(features.begin(), features.end(),
                features_vec.data() + i * chunk_size * feature_dim);

      results[i] = std::move(ss[i]->GetResult());
      states_vec[i] = std::move(ss[i]->GetStates());
      all_processed_frames[i] = num_processed_frames;
    }

    auto memory_info =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

    std::array<int64_t, 3> x_shape{n, chunk_size, feature_dim};

    Ort::Value x = Ort::Value::CreateTensor(memory_info, features_vec.data(),
                                            features_vec.size(), x_shape.data(),
                                            x_shape.size());

    std::array<int64_t, 1> processed_frames_shape{
      static_cast<int64_t>(all_processed_frames.size())};

    Ort::Value processed_frames = Ort::Value::CreateTensor(
      memory_info,
      all_processed_frames.data(),
      all_processed_frames.size(),
      processed_frames_shape.data(),
      processed_frames_shape.size());

    auto states = model_->StackStates(states_vec);

    auto pair = model_->RunEncoder(
      std::move(x), std::move(states), std::move(processed_frames));

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

    // TODO(fangjun): Remember to change these constants if needed
    int32_t frame_shift_ms = 10;
    int32_t subsampling_factor = 4;
    return Convert(decoder_result, sym_, frame_shift_ms, subsampling_factor);
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
    // we keep the decoder_out
    decoder_->UpdateDecoderOut(&s->GetResult());
    Ort::Value decoder_out = std::move(s->GetResult().decoder_out);
    s->SetResult(decoder_->GetEmptyResult());
    s->GetResult().decoder_out = std::move(decoder_out);

    // Note: We only update counters. The underlying audio samples
    // are not discarded.
    s->Reset();
  }

 private:
  OnlineRecognizerConfig config_;
  std::unique_ptr<OnlineTransducerModel> model_;
  std::unique_ptr<OnlineLM> lm_;
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
