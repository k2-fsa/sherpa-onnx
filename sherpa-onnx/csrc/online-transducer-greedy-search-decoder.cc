// sherpa-onnx/csrc/online-transducer-greedy-search-decoder.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/online-transducer-greedy-search-decoder.h"

#include <assert.h>

#include <algorithm>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/onnx-utils.h"

namespace sherpa_onnx {

static Ort::Value GetFrame(Ort::Value *encoder_out, int32_t t) {
  std::vector<int64_t> encoder_out_shape =
      encoder_out->GetTensorTypeAndShapeInfo().GetShape();
  assert(encoder_out_shape[0] == 1);

  int32_t encoder_out_dim = encoder_out_shape[2];

  auto memory_info =
      Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

  std::array<int64_t, 2> shape{1, encoder_out_dim};

  return Ort::Value::CreateTensor(
      memory_info,
      encoder_out->GetTensorMutableData<float>() + t * encoder_out_dim,
      encoder_out_dim, shape.data(), shape.size());
}

static Ort::Value Repeat(OrtAllocator *allocator, Ort::Value *cur_encoder_out,
                         int32_t n) {
  if (n == 1) {
    return std::move(*cur_encoder_out);
  }

  std::vector<int64_t> cur_encoder_out_shape =
      cur_encoder_out->GetTensorTypeAndShapeInfo().GetShape();

  std::array<int64_t, 2> ans_shape{n, cur_encoder_out_shape[1]};

  Ort::Value ans = Ort::Value::CreateTensor<float>(allocator, ans_shape.data(),
                                                   ans_shape.size());

  const float *src = cur_encoder_out->GetTensorData<float>();
  float *dst = ans.GetTensorMutableData<float>();
  for (int32_t i = 0; i != n; ++i) {
    std::copy(src, src + cur_encoder_out_shape[1], dst);
    dst += cur_encoder_out_shape[1];
  }

  return ans;
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
    fprintf(stderr,
            "Size mismatch! encoder_out.size(0) %d, result.size(0): %d\n",
            static_cast<int32_t>(encoder_out_shape[0]),
            static_cast<int32_t>(result->size()));
    exit(-1);
  }

  int32_t batch_size = static_cast<int32_t>(encoder_out_shape[0]);
  int32_t num_frames = static_cast<int32_t>(encoder_out_shape[1]);
  int32_t vocab_size = model_->VocabSize();

  Ort::Value decoder_input = model_->BuildDecoderInput(*result);
  Ort::Value decoder_out = model_->RunDecoder(std::move(decoder_input));

  for (int32_t t = 0; t != num_frames; ++t) {
    Ort::Value cur_encoder_out = GetFrame(&encoder_out, t);
    cur_encoder_out = Repeat(model_->Allocator(), &cur_encoder_out, batch_size);
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
      decoder_input = model_->BuildDecoderInput(*result);
      decoder_out = model_->RunDecoder(std::move(decoder_input));
    }
  }
}

}  // namespace sherpa_onnx
