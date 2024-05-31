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
  r.tokens.resize(context_size, -1);
  r.tokens.back() = blank_id;

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

  if (encoder_out_shape[0] != static_cast<int32_t>(result->size())) {
    SHERPA_ONNX_LOGE(
        "Size mismatch! encoder_out.size(0) %d, result.size(0): %d",
        static_cast<int32_t>(encoder_out_shape[0]),
        static_cast<int32_t>(result->size()));
    exit(-1);
  }

  int32_t batch_size = static_cast<int32_t>(encoder_out_shape[0]);
  int32_t num_frames = static_cast<int32_t>(encoder_out_shape[1]);
  int32_t vocab_size = model_->VocabSize();

  Ort::Value decoder_out{nullptr};
  bool is_batch_decoder_out_cached = true;
  for (const auto &r : *result) {
    if (!r.decoder_out) {
      is_batch_decoder_out_cached = false;
      break;
    }
  }

  if (is_batch_decoder_out_cached) {
    auto &r = result->front();
    std::vector<int64_t> decoder_out_shape =
        r.decoder_out.GetTensorTypeAndShapeInfo().GetShape();
    decoder_out_shape[0] = batch_size;
    decoder_out = Ort::Value::CreateTensor<float>(model_->Allocator(),
                                                  decoder_out_shape.data(),
                                                  decoder_out_shape.size());
    UseCachedDecoderOut(*result, &decoder_out);
  } else {
    Ort::Value decoder_input = model_->BuildDecoderInput(*result);
    decoder_out = model_->RunDecoder(std::move(decoder_input));
  }

  for (int32_t t = 0; t != num_frames; ++t) {
    Ort::Value cur_encoder_out =
        GetEncoderOutFrame(model_->Allocator(), &encoder_out, t);
    Ort::Value logit =
        model_->RunJoiner(std::move(cur_encoder_out), View(&decoder_out));

    float *p_logit = logit.GetTensorMutableData<float>();

    bool emitted = false;
    for (int32_t i = 0; i < batch_size; ++i, p_logit += vocab_size) {
      auto &r = (*result)[i];
      if (blank_penalty_ > 0.0) {
        p_logit[0] -= blank_penalty_;  // assuming blank id is 0
      }

      auto y = static_cast<int32_t>(std::distance(
          static_cast<const float *>(p_logit),
          std::max_element(static_cast<const float *>(p_logit),
                           static_cast<const float *>(p_logit) + vocab_size)));
      // blank id is hardcoded to 0
      // also, it treats unk as blank
      if (y != 0 && y != unk_id_) {
        emitted = true;
        r.tokens.push_back(y);
        r.timestamps.push_back(t + r.frame_offset);
        r.num_trailing_blanks = 0;
      } else {
        ++r.num_trailing_blanks;
      }

      // export the per-token log scores
      if (y != 0 && y != unk_id_) {
        // apply temperature-scaling
        for (int32_t n = 0; n < vocab_size; ++n) {
          p_logit[n] /= temperature_scale_;
        }
        LogSoftmax(p_logit, vocab_size);   // renormalize probabilities,
                                           // save time by doing it only for
                                           // emitted symbols
        const float *p_logprob = p_logit;  // rename p_logit as p_logprob,
                                           // now it contains normalized
                                           // probability
        r.ys_probs.push_back(p_logprob[y]);
      }
    }
    if (emitted) {
      Ort::Value decoder_input = model_->BuildDecoderInput(*result);
      decoder_out = model_->RunDecoder(std::move(decoder_input));
    }
  }

  UpdateCachedDecoderOut(model_->Allocator(), &decoder_out, result);

  // Update frame_offset
  for (auto &r : *result) {
    r.frame_offset += num_frames;
  }
}

}  // namespace sherpa_onnx
