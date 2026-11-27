// sherpa-onnx/csrc/online-recognizer-gigaam-v4-impl.cc
//
// Copyright (c)  2026  sherpa-onnx contributors

#include "sherpa-onnx/csrc/online-recognizer-gigaam-v4-impl.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/online-recognizer-impl.h"
#include "sherpa-onnx/csrc/onnx-utils.h"
#include "sherpa-onnx/csrc/session.h"
#include "sherpa-onnx/csrc/symbol-table.h"
#include "sherpa-onnx/csrc/text-utils.h"

namespace sherpa_onnx {
namespace {

constexpr int32_t kMelBins = 64;
constexpr int32_t kChunk = 64;  // 640 ms in 10 ms mel frames
constexpr int32_t kSubsampling = 4;
constexpr int32_t kEncoderDim = 768;
constexpr int32_t kPredictionDim = 320;
constexpr int32_t kBlank = 1024;
constexpr int32_t kMaxSymbols = 10;

FeatureExtractorConfig GigaAmFeatures() {
  FeatureExtractorConfig c;
  c.feature_dim = kMelBins;
  c.low_freq = 0;
  c.high_freq = 8000;
  c.snip_edges = true;
  c.frame_length_ms = 20;
  c.remove_dc_offset = false;
  c.preemph_coeff = 0;
  c.window_type = "hann";
  c.round_to_power_of_two = false;
  c.is_gigaam = true;
  return c;
}

std::string FrontendPath(const std::string &encoder) {
  for (const std::string suffix :
       {"_stream_step.int8.onnx", "_stream_step.int4.onnx",
        "_stream_step.fp16.onnx", "_stream_step.onnx"}) {
    if (EndsWith(encoder, suffix)) {
      return encoder.substr(0, encoder.size() - suffix.size()) +
             "_stream_frontend.onnx";
    }
  }

  throw std::invalid_argument(
      "GigaAM-v4 encoder must be named *_stream_step[.int8|.int4|.fp16].onnx");
}

enum class Head { kUnknown, kE2e, kRaw };

Head NamedHead(const std::string &path, const std::string &e2e,
               const std::string &raw) {
  if (EndsWith(path, e2e)) return Head::kE2e;
  if (EndsWith(path, raw)) return Head::kRaw;
  return Head::kUnknown;
}

bool IsRawModel(const OnlineModelConfig &config) {
  if (!config.tokens_buf.empty()) {
    throw std::invalid_argument(
        "GigaAM-v4 requires a token file matching the decoder head; "
        "tokens_buf is unsupported");
  }
  const auto tokens =
      NamedHead(config.tokens, "tokens.e2e.txt", "tokens.raw.txt");
  const auto decoder = NamedHead(config.transducer.decoder, "_e2e_decoder.onnx",
                                 "_raw_decoder.onnx");
  const auto joint =
      NamedHead(config.transducer.joiner, "_e2e_joint.onnx", "_raw_joint.onnx");
  Head head = Head::kUnknown;
  for (const auto candidate : {decoder, joint, tokens}) {
    if (candidate == Head::kUnknown) continue;
    if (head != Head::kUnknown && head != candidate) {
      throw std::invalid_argument(
          "GigaAM-v4 decoder, joint, and tokens must use the same head");
    }
    head = candidate;
  }
  return head == Head::kRaw;
}

int64_t ReadEncoderLength(const Ort::Value &length) {
  if (!length.IsTensor()) {
    throw std::runtime_error("Unexpected GigaAM-v4 encoder length tensor");
  }
  const auto info = length.GetTensorTypeAndShapeInfo();
  if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 ||
      info.GetShape() != std::vector<int64_t>{1}) {
    throw std::runtime_error("Unexpected GigaAM-v4 encoder length tensor");
  }
  return length.GetTensorData<int64_t>()[0];
}

float *CheckedFloatData(Ort::Value *tensor, size_t count,
                        const char *diagnostic) {
  if (!tensor->IsTensor()) {
    throw std::runtime_error(diagnostic);
  }
  const auto info = tensor->GetTensorTypeAndShapeInfo();
  if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
      info.GetElementCount() != count) {
    throw std::runtime_error(diagnostic);
  }
  return tensor->GetTensorMutableData<float>();
}

class Session {
 public:
  Session(Ort::Env &env, const std::string &path,
          const Ort::SessionOptions &options, size_t inputs, size_t outputs)
      : session_(env, SHERPA_ONNX_TO_ORT_PATH(path), options) {
    if (session_.GetInputCount() != inputs ||
        session_.GetOutputCount() != outputs) {
      throw std::invalid_argument("Unexpected GigaAM-v4 ONNX graph: " + path);
    }
    GetInputNames(&session_, &input_names_, &input_ptrs_);
    GetOutputNames(&session_, &output_names_, &output_ptrs_);
  }

  std::vector<Ort::Value> ZeroState(size_t start) const {
    Ort::AllocatorWithDefaultOptions allocator;
    std::vector<Ort::Value> state;
    for (size_t i = start; i < input_ptrs_.size(); ++i) {
      auto type = session_.GetInputTypeInfo(i);
      auto info = type.GetTensorTypeAndShapeInfo();
      auto shape = info.GetShape();
      // The predictor's batch axis is symbolic even in the one-stream export.
      for (auto &dim : shape) {
        if (dim < 0) dim = 1;
      }
      if (info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        auto value = Ort::Value::CreateTensor<float>(allocator, shape.data(),
                                                     shape.size());
        Fill(&value, 0.0f);
        state.push_back(std::move(value));
      } else if (info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
        auto value = Ort::Value::CreateTensor<int64_t>(allocator, shape.data(),
                                                       shape.size());
        Fill(&value, int64_t{0});
        state.push_back(std::move(value));
      } else {
        throw std::invalid_argument("Unsupported GigaAM-v4 state tensor type");
      }
    }
    return state;
  }

  std::vector<Ort::Value> Run(const std::vector<Ort::Value> &inputs) const {
    return session_.Run({}, input_ptrs_.data(), inputs.data(), inputs.size(),
                        output_ptrs_.data(), output_ptrs_.size());
  }

 private:
  mutable Ort::Session session_;
  std::vector<std::string> input_names_, output_names_;
  std::vector<const char *> input_ptrs_, output_ptrs_;
};

template <typename T, size_t N>
Ort::Value Tensor(T *data, size_t count, const std::array<int64_t, N> &shape) {
  auto memory =
      Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
  return Ort::Value::CreateTensor<T>(memory, data, count, shape.data(), N);
}

void AppendState(std::vector<Ort::Value> *destination,
                 std::vector<Ort::Value> *state, size_t start = 0) {
  for (size_t i = start; i < state->size(); ++i) {
    destination->push_back(View(&(*state)[i]));
  }
}

void KeepState(std::vector<Ort::Value> *state, std::vector<Ort::Value> *outputs,
               size_t start) {
  state->assign(std::make_move_iterator(outputs->begin() + start),
                std::make_move_iterator(outputs->end()));
}

struct Stream : OnlineStream {
  explicit Stream(const FeatureExtractorConfig &config)
      : OnlineStream(config) {}

  std::vector<Ort::Value> frontend, encoder, predictor;
  int32_t encoded_frames = 0;
};

class Recognizer : public OnlineRecognizerImpl {
 public:
  Recognizer(const OnlineRecognizerConfig &config, bool raw)
      : OnlineRecognizerImpl(config),
        options_(GetSessionOptions(config.model_config)),
        frontend_(env_, FrontendPath(config.model_config.transducer.encoder),
                  options_, 6, 6),
        encoder_(env_, config.model_config.transducer.encoder, options_, 58,
                 58),
        decoder_(env_, config.model_config.transducer.decoder, options_, 3, 3),
        joiner_(env_, config.model_config.transducer.joiner, options_, 2, 1),
        symbols_(config.model_config.tokens),
        endpoint_(config.endpoint_config),
        enable_endpoint_(config.enable_endpoint),
        blank_penalty_(config.blank_penalty),
        raw_(raw) {
    if (!symbols_.Contains("<blk>") || symbols_["<blk>"] != kBlank ||
        symbols_.NumSymbols() != kBlank + 1) {
      throw std::invalid_argument(
          "GigaAM-v4 tokens must have 1024 pieces followed by <blk> 1024");
    }
    for (int32_t id = 0; id < kBlank; ++id) {
      if (!symbols_.Contains(id)) {
        throw std::invalid_argument("Missing ID in GigaAM-v4 token table");
      }
    }
  }

  std::unique_ptr<OnlineStream> CreateStream() const override {
    return std::make_unique<Stream>(GigaAmFeatures());
  }

  bool IsReady(OnlineStream *s) const override {
    const int32_t ready = s->NumFramesReady();
    const int32_t remaining = ready - s->GetNumProcessedFrames();
    return remaining >= kChunk || (remaining > 0 && s->IsLastFrame(ready - 1));
  }

  void DecodeStreams(OnlineStream **streams, int32_t n) const override {
    for (int32_t i = 0; i != n; ++i) {
      if (IsReady(streams[i])) Decode(static_cast<Stream *>(streams[i]));
    }
  }

  OnlineRecognizerResult GetResult(OnlineStream *stream) const override {
    auto *s = static_cast<Stream *>(stream);
    const auto &decoded = s->GetResult();
    OnlineRecognizerResult r;
    r.segment = s->GetCurrentSegment();
    r.start_time = 0.01f * s->GetNumFramesSinceStart();
    r.num_trailing_blanks = decoded.num_trailing_blanks;
    r.tokens.reserve(decoded.tokens.size());
    r.timestamps.reserve(decoded.tokens.size());
    for (size_t i = 0; i < decoded.tokens.size(); ++i) {
      auto piece = symbols_[decoded.tokens[i]];
      if (piece == "<unk>") piece = raw_ ? " " : " ⁇ ";
      r.text += piece;
      r.tokens.push_back(std::move(piece));
      r.timestamps.push_back(decoded.timestamps[i] * 0.04f);
    }
    r.text = ApplyInverseTextNormalization(Trim(r.text));
    r.text = ApplyHomophoneReplacer(std::move(r.text));
    return r;
  }

  bool IsEndpoint(OnlineStream *stream) const override {
    auto *s = static_cast<Stream *>(stream);
    return enable_endpoint_ &&
           endpoint_.IsEndpoint(
               s->GetNumProcessedFrames(),
               s->GetResult().num_trailing_blanks * kSubsampling, 0.01f);
  }

  void Reset(OnlineStream *stream) const override {
    auto *s = static_cast<Stream *>(stream);
    if (!s->GetResult().tokens.empty()) ++s->GetCurrentSegment();
    if (IsEndpoint(s)) {
      s->Reset();  // Keep queued frames for the next segment.
    } else {
      s->ResetFeatures();
    }
    s->frontend.clear();
    s->encoder.clear();
    s->predictor.clear();
    s->SetResult({});
    s->encoded_frames = 0;
  }

 private:
  void Decode(Stream *s) const {
    auto &processed = s->GetNumProcessedFrames();
    const int32_t remaining = s->NumFramesReady() - processed;
    const int32_t complete = remaining / kChunk * kChunk;
    // Drain complete chunks before padding the EOF remainder.
    int64_t count = complete > 0 ? complete : remaining;
    const int64_t padded = (count + kChunk - 1) / kChunk * kChunk;
    auto frames = s->GetFrames(processed, static_cast<int32_t>(count));
    std::vector<float> mel(kMelBins * padded);
    for (int32_t t = 0; t < count; ++t) {
      for (int32_t bin = 0; bin < kMelBins; ++bin) {
        mel[bin * padded + t] = frames[t * kMelBins + bin];
      }
    }

    if (s->frontend.empty()) {
      s->frontend = frontend_.ZeroState(2);
      s->encoder = encoder_.ZeroState(2);
    }

    // EOF pads the physical chunk, while both graphs see the true length.
    std::vector<Ort::Value> inputs;
    inputs.push_back(Tensor(mel.data(), mel.size(),
                            std::array<int64_t, 3>{1, kMelBins, padded}));
    inputs.push_back(Tensor(&count, 1, std::array<int64_t, 1>{1}));
    AppendState(&inputs, &s->frontend);
    auto front = frontend_.Run(inputs);
    KeepState(&s->frontend, &front, 2);

    inputs.clear();
    inputs.push_back(View(&front[0]));
    inputs.push_back(View(&front[1]));
    AppendState(&inputs, &s->encoder);
    auto encoded = encoder_.Run(inputs);
    KeepState(&s->encoder, &encoded, 2);
    DecodeEncoderOutput(encoded[0], ReadEncoderLength(encoded[1]), s);
    processed += static_cast<int32_t>(count);
  }

  void Predict(int64_t token, Stream *s) const {
    if (s->predictor.empty()) {
      s->predictor.emplace_back(nullptr);
      auto state = decoder_.ZeroState(1);
      for (auto &value : state) s->predictor.push_back(std::move(value));
    }
    std::vector<Ort::Value> inputs;
    inputs.push_back(Tensor(&token, 1, std::array<int64_t, 2>{1, 1}));
    AppendState(&inputs, &s->predictor, 1);
    s->predictor = decoder_.Run(inputs);
  }

  int32_t Join(std::array<float, kEncoderDim> *frame, Stream *s) const {
    std::vector<Ort::Value> inputs;
    inputs.push_back(Tensor(frame->data(), frame->size(),
                            std::array<int64_t, 3>{1, kEncoderDim, 1}));
    auto *prediction =
        CheckedFloatData(&s->predictor.front(), kPredictionDim,
                         "Unexpected GigaAM-v4 predictor output");
    inputs.push_back(Tensor(prediction, kPredictionDim,
                            std::array<int64_t, 3>{1, kPredictionDim, 1}));
    auto scores = joiner_.Run(inputs);
    auto *logits = CheckedFloatData(&scores[0], kBlank + 1,
                                    "Unexpected GigaAM-v4 joint output");
    logits[kBlank] -= blank_penalty_;
    return std::max_element(logits, logits + kBlank + 1) - logits;
  }

  void DecodeFrame(std::array<float, kEncoderDim> *frame, int32_t timestamp,
                   Stream *s) const {
    if (s->predictor.empty()) Predict(kBlank, s);
    auto &result = s->GetResult();
    for (int32_t n = 0; n < kMaxSymbols; ++n) {
      const int32_t token = Join(frame, s);
      if (token == kBlank) {
        ++result.num_trailing_blanks;
        break;
      }
      result.tokens.push_back(token);
      result.timestamps.push_back(timestamp);
      result.num_trailing_blanks = 0;
      Predict(token, s);
    }
  }

  void DecodeEncoderOutput(const Ort::Value &encoded, int64_t valid,
                           Stream *s) const {
    if (!encoded.IsTensor()) {
      throw std::runtime_error("Unexpected GigaAM-v4 encoder output");
    }
    const auto info = encoded.GetTensorTypeAndShapeInfo();
    const auto shape = info.GetShape();
    if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
        shape.size() != 3 || shape[0] != 1 || shape[1] != kEncoderDim ||
        valid < 0 || valid > shape[2]) {
      throw std::runtime_error("Unexpected GigaAM-v4 encoder output");
    }
    const auto *data = encoded.GetTensorData<float>();
    for (int32_t i = 0; i < valid; ++i) {
      std::array<float, kEncoderDim> frame;
      for (int32_t d = 0; d < kEncoderDim; ++d) {
        frame[d] = data[d * shape[2] + i];
      }
      DecodeFrame(&frame, s->encoded_frames + i, s);
    }
    s->encoded_frames += static_cast<int32_t>(valid);
  }

  Ort::Env env_{ORT_LOGGING_LEVEL_ERROR};
  Ort::SessionOptions options_;
  Session frontend_, encoder_, decoder_, joiner_;
  SymbolTable symbols_;
  Endpoint endpoint_;
  bool enable_endpoint_;
  float blank_penalty_;
  bool raw_;
};

}  // namespace

std::unique_ptr<OnlineRecognizerImpl> CreateOnlineGigaAmV4(
    const OnlineRecognizerConfig &config) {
  if (config.decoding_method != "greedy_search" ||
      config.model_config.provider_config.provider != "cpu") {
    throw std::invalid_argument(
        "GigaAM-v4 online supports the CPU provider and greedy_search");
  }
  return std::make_unique<Recognizer>(config, IsRawModel(config.model_config));
}

}  // namespace sherpa_onnx
