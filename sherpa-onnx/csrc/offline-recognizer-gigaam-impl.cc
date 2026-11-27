// sherpa-onnx/csrc/offline-recognizer-gigaam-impl.cc
//
// Copyright (c)  2026  sherpa-onnx contributors

#include "sherpa-onnx/csrc/offline-recognizer-gigaam-impl.h"

#include <algorithm>
#include <array>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/gigaam-features.h"
#include "sherpa-onnx/csrc/gigaam-sentencepiece-decoder.h"
#include "sherpa-onnx/csrc/offline-recognizer-impl.h"
#include "sherpa-onnx/csrc/onnx-utils.h"
#include "sherpa-onnx/csrc/session.h"
#include "sherpa-onnx/csrc/text-utils.h"

namespace sherpa_onnx {

namespace {

constexpr std::array<const char *, 3> kEncoderInputs{"audio_signal", "length",
                                                     "language_vector"};
constexpr std::array<const char *, 2> kEncoderOutputs{"head_outputs",
                                                      "encoded_len"};
constexpr std::array<const char *, 2> kAdvanceInputs{"projected", "lengths"};
constexpr std::array<const char *, 1> kAdvanceOutputs{"tokens"};
constexpr int32_t kNumLanguages = 32;
constexpr int32_t kNumHeads = 2;
constexpr int32_t kProjectionDim = 320;
constexpr int32_t kMaxDurationSeconds = 30;

template <size_t N, size_t M>
void CheckGraph(Ort::Session *session,
                const std::array<const char *, N> &inputs, size_t num_inputs,
                const std::array<const char *, M> &outputs) {
  std::vector<std::string> input_names;
  std::vector<std::string> output_names;
  std::vector<const char *> input_ptrs;
  std::vector<const char *> output_ptrs;
  GetInputNames(session, &input_names, &input_ptrs);
  GetOutputNames(session, &output_names, &output_ptrs);
  if (input_names.size() != num_inputs ||
      !std::equal(input_names.begin(), input_names.end(), inputs.begin()) ||
      !std::is_permutation(output_names.begin(), output_names.end(),
                           outputs.begin(), outputs.end())) {
    throw std::invalid_argument("Unexpected GigaAM RNN-T graph inputs/outputs");
  }
}

int64_t ProjectionFrameCount(const Ort::Value &projected) {
  if (!projected.IsTensor()) {
    throw std::runtime_error("Unexpected GigaAM encoder projection tensor");
  }
  const auto info = projected.GetTensorTypeAndShapeInfo();
  const auto shape = info.GetShape();
  if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
      shape.size() != 4 || shape[0] != kNumHeads || shape[1] != 1 ||
      shape[3] != kProjectionDim) {
    throw std::runtime_error("Unexpected GigaAM encoder projection tensor");
  }
  return shape[2];
}

void CheckEncodedLength(const Ort::Value &lengths, int64_t frames) {
  if (!lengths.IsTensor()) {
    throw std::runtime_error("Unexpected GigaAM encoder lengths tensor");
  }
  const auto info = lengths.GetTensorTypeAndShapeInfo();
  if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 ||
      info.GetShape() != std::vector<int64_t>{1}) {
    throw std::runtime_error("Unexpected GigaAM encoder lengths tensor");
  }
  const int64_t length = lengths.GetTensorData<int64_t>()[0];
  if (length < 0 || length > frames) {
    throw std::runtime_error("GigaAM encoder length is outside projection");
  }
}

class OfflineRecognizerGigaAMImpl : public OfflineRecognizerImpl {
 public:
  explicit OfflineRecognizerGigaAMImpl(const OfflineRecognizerConfig &config)
      : OfflineRecognizerImpl(config),
        tokenizer_(config.model_config.gigaam.tokenizer),
        head_(config.model_config.gigaam.head == "e2e" ? 0 : 1),
        env_(ORT_LOGGING_LEVEL_ERROR) {
    const auto &model = config.model_config.gigaam;
    auto opts = GetSessionOptions(config.model_config);
    opts.AddConfigEntry("session.intra_op.allow_spinning", "0");

    encoder_ = std::make_unique<Ort::Session>(
        env_, SHERPA_ONNX_TO_ORT_PATH(model.encoder), opts);
    const size_t num_inputs = encoder_->GetInputCount();
    has_language_input_ = num_inputs == kEncoderInputs.size();
    CheckGraph(encoder_.get(), kEncoderInputs, has_language_input_ ? 3 : 2,
               kEncoderOutputs);
    LanguageSlot(model.language);

    advance_ = std::make_unique<Ort::Session>(
        env_, SHERPA_ONNX_TO_ORT_PATH(model.advance), opts);
    CheckGraph(advance_.get(), kAdvanceInputs, kAdvanceInputs.size(),
               kAdvanceOutputs);
  }

  std::unique_ptr<OfflineStream> CreateStream() const override {
    return std::make_unique<OfflineStream>(
        SpeakerDiarizationTag{GigaAMFeatureExtractor::kSampleRate});
  }

  void DecodeStreams(OfflineStream **streams, int32_t n) const override {
    for (int32_t i = 0; i < n; ++i) {
      DecodeStream(streams[i]);
    }
  }

  OfflineRecognizerConfig GetConfig() const override { return config_; }

 private:
  int32_t LanguageSlot(const std::string &language) const {
    const int32_t slot = GigaAMLanguageSlot(language);
    if (slot < 0) {
      throw std::invalid_argument("Unknown GigaAM language: " + language);
    }
    if (!has_language_input_ && slot != 0) {
      throw std::invalid_argument(
          "The GigaAM-v4 encoder does not accept a language hint");
    }
    return slot;
  }

  void DecodeStream(OfflineStream *stream) const {
    const auto &language = stream->HasOption("language")
                               ? stream->GetOption("language")
                               : config_.model_config.gigaam.language;
    const int32_t slot = LanguageSlot(language);
    const auto samples = stream->GetFrames();
    if (samples.size() >
        GigaAMFeatureExtractor::kSampleRate * kMaxDurationSeconds) {
      throw std::invalid_argument("GigaAM expects utterances up to 30 seconds");
    }

    auto features = features_.Compute(samples);
    if (features.empty()) {
      stream->SetResult({});
      return;
    }

    const auto tokens = RunGraphs(&features, slot);
    stream->SetResult(DecodeTokens(tokens));
  }

  Ort::Value RunGraphs(std::vector<float> *features, int32_t slot) const {
    const auto memory =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
    constexpr int32_t kBins = GigaAMFeatureExtractor::kNumBins;
    int64_t num_frames = static_cast<int64_t>(features->size() / kBins);
    const std::array<int64_t, 3> feature_shape{1, kBins, num_frames};
    constexpr std::array<int64_t, 1> length_shape{1};
    std::array<Ort::Value, 3> inputs{
        Ort::Value::CreateTensor(memory, features->data(), features->size(),
                                 feature_shape.data(), feature_shape.size()),
        Ort::Value::CreateTensor(memory, &num_frames, 1, length_shape.data(),
                                 length_shape.size()),
        Ort::Value{nullptr}};

    std::array<float, kNumLanguages> language_vector{};
    if (has_language_input_) {
      language_vector[slot] = 1.0f;
      constexpr std::array<int64_t, 2> shape{1, kNumLanguages};
      inputs[2] = Ort::Value::CreateTensor(memory, language_vector.data(),
                                           language_vector.size(), shape.data(),
                                           shape.size());
    }

    auto encoded = encoder_->Run(
        {}, kEncoderInputs.data(), inputs.data(), has_language_input_ ? 3 : 2,
        kEncoderOutputs.data(), kEncoderOutputs.size());
    CheckEncodedLength(encoded[1], ProjectionFrameCount(encoded[0]));
    auto output =
        advance_->Run({}, kAdvanceInputs.data(), encoded.data(), encoded.size(),
                      kAdvanceOutputs.data(), kAdvanceOutputs.size());
    return std::move(output[0]);
  }

  OfflineRecognitionResult DecodeTokens(const Ort::Value &tokens) const {
    if (!tokens.IsTensor()) {
      throw std::runtime_error("Unexpected GigaAM advance token tensor");
    }
    const auto info = tokens.GetTensorTypeAndShapeInfo();
    const auto shape = info.GetShape();
    if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 ||
        shape.size() != 3 || shape[1] != kNumHeads || shape[2] != 1) {
      throw std::runtime_error("Unexpected GigaAM advance token tensor");
    }

    const auto *data = tokens.GetTensorData<int64_t>();
    std::vector<int64_t> ids;
    OfflineRecognitionResult result;
    for (int64_t i = 0; i < shape[0]; ++i) {
      const int64_t id = data[i * kNumHeads + head_];
      if (id >= 0) {
        ids.push_back(id);
        result.tokens.push_back(tokenizer_.Piece(id));
      }
    }
    result.text = tokenizer_.Decode(ids);
    result.text = ApplyInverseTextNormalization(result.text);
    result.text = ApplyHomophoneReplacer(result.text);
    return result;
  }

  GigaAMFeatureExtractor features_;
  GigaAMSentencePieceDecoder tokenizer_;
  int32_t head_;
  bool has_language_input_ = false;
  Ort::Env env_;
  std::unique_ptr<Ort::Session> encoder_;
  std::unique_ptr<Ort::Session> advance_;
};

}  // namespace

std::unique_ptr<OfflineRecognizerImpl> CreateOfflineRecognizerGigaAM(
    const OfflineRecognizerConfig &config) {
  if (config.model_config.provider != "cpu" ||
      config.model_config.num_threads < 1 ||
      config.decoding_method != "greedy_search" ||
      !config.model_config.gigaam.Validate()) {
    throw std::invalid_argument(
        "GigaAM requires valid model files, CPU and greedy_search");
  }
  return std::make_unique<OfflineRecognizerGigaAMImpl>(config);
}

}  // namespace sherpa_onnx
