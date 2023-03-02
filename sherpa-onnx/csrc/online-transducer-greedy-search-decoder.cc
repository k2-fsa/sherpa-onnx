// sherpa-onnx/csrc/online-transducer-greedy-search-decoder.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/online-transducer-greedy-search-decoder.h"

#include <algorithm>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/onnx-utils.h"

namespace sherpa_onnx {

static void UseCachedDecoderOut(
    const std::vector<OnlineTransducerDecoderResult> &results,
    Ort::Value *decoder_out) {
  std::vector<int64_t> shape =
      decoder_out->GetTensorTypeAndShapeInfo().GetShape();
  float *dst = decoder_out->GetTensorMutableData<float>();
  for (const auto &r : results) {
    if (r.decoder_out) {
      const float *src = r.decoder_out.GetTensorData<float>();
      std::copy(src, src + shape[1], dst);
    }
    dst += shape[1];
  }
}

static void UpdateCachedDecoderOut(
    OrtAllocator *allocator, const Ort::Value *decoder_out,
    std::vector<OnlineTransducerDecoderResult> *results) {
  std::vector<int64_t> shape =
      decoder_out->GetTensorTypeAndShapeInfo().GetShape();
  auto memory_info =
      Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
  std::array<int64_t, 2> v_shape{1, shape[1]};

  const float *src = decoder_out->GetTensorData<float>();
  for (auto &r : *results) {
    if (!r.decoder_out) {
      r.decoder_out = Ort::Value::CreateTensor<float>(allocator, v_shape.data(),
                                                      v_shape.size());
    }

    float *dst = r.decoder_out.GetTensorMutableData<float>();
    std::copy(src, src + shape[1], dst);
    src += shape[1];
  }
}

OnlineTransducerDecoderResult
OnlineTransducerGreedySearchDecoder::GetEmptyResult() const {
  int32_t context_size = model_->ContextSize();
  int32_t blank_id = 0;  // always 0
  OnlineTransducerDecoderResult r;
  r.tokens.resize(context_size, blank_id);

  return r;
}

void OnlineTransducerGreedySearchDecoder::StripLeadingBlanks(
    OnlineTransducerDecoderResult *r) const {
  int32_t context_size = model_->ContextSize();

  auto start = r->tokens.begin() + context_size;
  auto end = r->tokens.end();

  r->tokens = std::vector<int64_t>(start, end);
}

void OnlineTransducerGreedySearchDecoder::Decode(
    Ort::Value encoder_out,
    std::vector<OnlineTransducerDecoderResult> *result) {
  std::vector<int64_t> encoder_out_shape =
      encoder_out.GetTensorTypeAndShapeInfo().GetShape();

  if (encoder_out_shape[0] != result->size()) {
    SHERPA_ONNX_LOGE(
        "Size mismatch! encoder_out.size(0) %d, result.size(0): %d",
        static_cast<int32_t>(encoder_out_shape[0]),
        static_cast<int32_t>(result->size()));
    exit(-1);
  }

  int32_t batch_size = static_cast<int32_t>(encoder_out_shape[0]);
  int32_t num_frames = static_cast<int32_t>(encoder_out_shape[1]);
  int32_t vocab_size = model_->VocabSize();

  Ort::Value decoder_input = model_->BuildDecoderInput(*result);
  Ort::Value decoder_out = model_->RunDecoder(std::move(decoder_input));
  UseCachedDecoderOut(*result, &decoder_out);

  for (int32_t t = 0; t != num_frames; ++t) {
    Ort::Value cur_encoder_out =
        GetEncoderOutFrame(model_->Allocator(), &encoder_out, t);
    Ort::Value logit = model_->RunJoiner(
        std::move(cur_encoder_out), Clone(model_->Allocator(), &decoder_out));

    const float *p_logit = logit.GetTensorData<float>();

    bool emitted = false;
    for (int32_t i = 0; i < batch_size; ++i, p_logit += vocab_size) {
      auto y = static_cast<int32_t>(std::distance(
          static_cast<const float *>(p_logit),
          std::max_element(static_cast<const float *>(p_logit),
                           static_cast<const float *>(p_logit) + vocab_size)));
      if (y != 0) {
        emitted = true;
        (*result)[i].tokens.push_back(y);
        (*result)[i].num_trailing_blanks = 0;
      } else {
        ++(*result)[i].num_trailing_blanks;
      }
    }
    if (emitted) {
      Ort::Value decoder_input = model_->BuildDecoderInput(*result);
      decoder_out = model_->RunDecoder(std::move(decoder_input));
    }
  }

  UpdateCachedDecoderOut(model_->Allocator(), &decoder_out, result);
}

}  // namespace sherpa_onnx
