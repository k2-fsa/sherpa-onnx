// sherpa/csrc/online-transducer-greedy-search-decoder.cc
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

  if (result->size() != 1) {
    fprintf(stderr, "only batch size == 1 is implemented. Given: %d",
            static_cast<int32_t>(result->size()));
    exit(-1);
  }

  auto &hyp = (*result)[0].tokens;

  int32_t num_frames = encoder_out_shape[1];
  int32_t vocab_size = model_->VocabSize();

  Ort::Value decoder_input = model_->BuildDecoderInput(hyp);
  Ort::Value decoder_out = model_->RunDecoder(std::move(decoder_input));

  for (int32_t t = 0; t != num_frames; ++t) {
    Ort::Value cur_encoder_out = GetFrame(&encoder_out, t);
    Ort::Value logit =
        model_->RunJoiner(std::move(cur_encoder_out), Clone(&decoder_out));
    const float *p_logit = logit.GetTensorData<float>();

    auto y = static_cast<int32_t>(std::distance(
        static_cast<const float *>(p_logit),
        std::max_element(static_cast<const float *>(p_logit),
                         static_cast<const float *>(p_logit) + vocab_size)));
    if (y != 0) {
      hyp.push_back(y);
      decoder_input = model_->BuildDecoderInput(hyp);
      decoder_out = model_->RunDecoder(std::move(decoder_input));
    }
  }
}

}  // namespace sherpa_onnx
