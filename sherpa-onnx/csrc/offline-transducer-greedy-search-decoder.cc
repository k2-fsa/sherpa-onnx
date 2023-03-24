// sherpa-onnx/csrc/offline-transducer-greedy-search-decoder.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-transducer-greedy-search-decoder.h"

#include <algorithm>
#include <iterator>
#include <utility>

#include "sherpa-onnx/csrc/onnx-utils.h"

namespace sherpa_onnx {

std::vector<OfflineTransducerDecoderResult>
OfflineTransducerGreedySearchDecoder::Decode(Ort::Value encoder_out,
                                             Ort::Value encoder_out_length) {
  std::vector<int64_t> encoder_out_shape =
      encoder_out.GetTensorTypeAndShapeInfo().GetShape();

  assert(encoder_out_shape.size() == 3);

  int32_t batch_size = static_cast<int32_t>(encoder_out_shape[0]);
  if (batch_size != 1) {
    fprintf(stderr, "TODO(fangjun): Support batch size > 1\n");
    exit(-1);
  }

  int32_t num_frames = static_cast<int32_t>(encoder_out_shape[1]);

  int32_t vocab_size = model_->VocabSize();
  int32_t context_size = model_->ContextSize();
  std::vector<OfflineTransducerDecoderResult> ans(1);
  for (auto &r : ans) {
    // 0 is the ID of the blank token
    r.tokens.resize(context_size, 0);
  }

  auto decoder_input = model_->BuildDecoderInput(ans);
  Ort::Value decoder_out = model_->RunDecoder(std::move(decoder_input));

  for (int32_t t = 0; t != num_frames; ++t) {
    Ort::Value cur_encoder_out =
        GetEncoderOutFrame(model_->Allocator(), &encoder_out, t);
    Ort::Value logit = model_->RunJoiner(
        std::move(cur_encoder_out), Clone(model_->Allocator(), &decoder_out));

    const float *p_logit = logit.GetTensorData<float>();
    // TODO(fangjun): Process batch_size > 1
    auto y = static_cast<int32_t>(std::distance(
        static_cast<const float *>(p_logit),
        std::max_element(static_cast<const float *>(p_logit),
                         static_cast<const float *>(p_logit) + vocab_size)));

    if (y != 0) {
      ans[0].tokens.push_back(y);
      ans[0].timestamps.push_back(t);
      Ort::Value decoder_input = model_->BuildDecoderInput(ans);
      decoder_out = model_->RunDecoder(std::move(decoder_input));
    }
  }

  for (auto &r : ans) {
    r.tokens = {r.tokens.begin() + context_size, r.tokens.end()};
  }
  return ans;
}

}  // namespace sherpa_onnx
