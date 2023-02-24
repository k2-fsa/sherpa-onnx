// sherpa-onnx/csrc/online-transducer-modified-beam-search-decoder.cc
//
// Copyright (c)  2023  Pingfeng Luo

#include "sherpa-onnx/csrc/online-transducer-modified-beam-search-decoder.h"

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

static void LogSoftmax(float *in, int32_t w, int32_t h) {
  for (int32_t i = 0; i != h; ++i) {
    LogSoftmax(in, w);
    in += w;
  }
}

OnlineTransducerDecoderResult
OnlineTransducerModifiedBeamSearchDecoder::GetEmptyResult() const {
  int32_t context_size = model_->ContextSize();
  int32_t blank_id = 0;  // always 0
  OnlineTransducerDecoderResult r;
  r.tokens.resize(context_size, blank_id);

  return r;
}

void OnlineTransducerModifiedBeamSearchDecoder::StripLeadingBlanks(
    OnlineTransducerDecoderResult *r) const {
  int32_t context_size = model_->ContextSize();

  auto start = r->tokens.begin() + context_size;
  auto end = r->tokens.end();

  r->tokens = std::vector<int64_t>(start, end);
}

void OnlineTransducerModifiedBeamSearchDecoder::Decode(
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

  std::vector<Hypotheses> cur;
  for (auto &r : *result) {
    cur.push_back(std::move(r.hyps));
  }
  std::vector<Hypothesis> prev;

  for (int32_t t = 0; t != num_frames; ++t) {
    // Due to merging paths with identical token sequences,
    // not all utterances have "num_active_paths" paths.
    int32_t num_hyps = 0;
    std::vector<int32_t> hyps_shape_split;
    hyps_shape_split.push_back(0);

    prev.clear();
    prev.reserve(num_hyps);
    for (auto &hyps : cur) {
      for (auto &h : hyps) {
        prev.push_back(std::move(h.second));
      }
      num_hyps += cur.size();
      hyps_shape_split.push_back(num_hyps);
    }
    cur.clear();
    cur.reserve(batch_size);

    Ort::Value decoder_input = model_->BuildDecoderInput(prev);
    Ort::Value decoder_out = model_->RunDecoder(std::move(decoder_input));

    Ort::Value cur_encoder_out = GetFrame(&encoder_out, t);
    cur_encoder_out = Repeat(model_->Allocator(), &cur_encoder_out, batch_size);
    Ort::Value logit =
        model_->RunJoiner(std::move(cur_encoder_out), Clone(&decoder_out));
    float *p_logit = const_cast<float *>(logit.GetTensorData<float>());

    for (int32_t b = 0; b < batch_size; ++b) {
      int32_t cur_stream_hyps_num = hyps_shape_split[b + 1] - hyps_shape_split[b];
      LogSoftmax(p_logit, vocab_size, cur_stream_hyps_num);
      auto topk =
        TopkIndex(p_logit, vocab_size * cur_stream_hyps_num , 4);

      Hypotheses hyps;
      for (auto i : topk) {
        int32_t hyp_index = i / vocab_size + hyps_shape_split[b];
        int32_t new_token = i % vocab_size;

        Hypothesis new_hyp = prev[hyp_index];
        if (new_token != 0) {
          new_hyp.ys.push_back(new_token);
          new_hyp.num_trailing_blanks = 0;
        } else {
          ++new_hyp.num_trailing_blanks;
        }
        new_hyp.log_prob += p_logit[i];
        hyps.Add(std::move(new_hyp));
      }
      cur.push_back(std::move(hyps));
      p_logit += vocab_size * cur_stream_hyps_num;
    }
  }

  for (int32_t i = 0; i != batch_size; ++i) {
    (*result)[i].hyps = std::move(cur[i]);
  }
}

}  // namespace sherpa_onnx
