// sherpa-onnx/csrc/online-transducer-modified-beam-search-decoder.cc
//
// Copyright (c)  2023  Pingfeng Luo
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/online-transducer-modified-beam-search-decoder.h"

#include <algorithm>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/onnx-utils.h"

namespace sherpa_onnx {

static void UseCachedDecoderOut(
    const std::vector<int32_t> &hyps_num_split,
    const std::vector<OnlineTransducerDecoderResult> &results,
    int32_t context_size, Ort::Value *decoder_out) {
  std::vector<int64_t> shape =
      decoder_out->GetTensorTypeAndShapeInfo().GetShape();

  float *dst = decoder_out->GetTensorMutableData<float>();

  int32_t batch_size = static_cast<int32_t>(results.size());
  for (int32_t i = 0; i != batch_size; ++i) {
    int32_t num_hyps = hyps_num_split[i + 1] - hyps_num_split[i];
    if (num_hyps > 1 || !results[i].decoder_out) {
      dst += num_hyps * shape[1];
      continue;
    }

    const float *src = results[i].decoder_out.GetTensorData<float>();
    std::copy(src, src + shape[1], dst);
    dst += shape[1];
  }
}

static Ort::Value Repeat(OrtAllocator *allocator, Ort::Value *cur_encoder_out,
                         const std::vector<int32_t> &hyps_num_split) {
  std::vector<int64_t> cur_encoder_out_shape =
      cur_encoder_out->GetTensorTypeAndShapeInfo().GetShape();

  std::array<int64_t, 2> ans_shape{hyps_num_split.back(),
                                   cur_encoder_out_shape[1]};

  Ort::Value ans = Ort::Value::CreateTensor<float>(allocator, ans_shape.data(),
                                                   ans_shape.size());

  const float *src = cur_encoder_out->GetTensorData<float>();
  float *dst = ans.GetTensorMutableData<float>();
  int32_t batch_size = static_cast<int32_t>(hyps_num_split.size()) - 1;
  for (int32_t b = 0; b != batch_size; ++b) {
    int32_t cur_stream_hyps_num = hyps_num_split[b + 1] - hyps_num_split[b];
    for (int32_t i = 0; i != cur_stream_hyps_num; ++i) {
      std::copy(src, src + cur_encoder_out_shape[1], dst);
      dst += cur_encoder_out_shape[1];
    }
    src += cur_encoder_out_shape[1];
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
  std::vector<int64_t> blanks(context_size, blank_id);
  Hypotheses blank_hyp({{blanks, 0}});
  r.hyps = std::move(blank_hyp);
  return r;
}

void OnlineTransducerModifiedBeamSearchDecoder::StripLeadingBlanks(
    OnlineTransducerDecoderResult *r) const {
  int32_t context_size = model_->ContextSize();
  auto hyp = r->hyps.GetMostProbable(true);

  std::vector<int64_t> tokens(hyp.ys.begin() + context_size, hyp.ys.end());
  r->tokens = std::move(tokens);
  r->num_trailing_blanks = hyp.num_trailing_blanks;
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
    int32_t hyps_num_acc = 0;
    std::vector<int32_t> hyps_num_split;
    hyps_num_split.push_back(0);

    prev.clear();
    for (auto &hyps : cur) {
      for (auto &h : hyps) {
        prev.push_back(std::move(h.second));
        hyps_num_acc++;
      }
      hyps_num_split.push_back(hyps_num_acc);
    }
    cur.clear();
    cur.reserve(batch_size);

    Ort::Value decoder_input = model_->BuildDecoderInput(prev);
    Ort::Value decoder_out = model_->RunDecoder(std::move(decoder_input));
    if (t == 0) {
      UseCachedDecoderOut(hyps_num_split, *result, model_->ContextSize(),
                          &decoder_out);
    }

    Ort::Value cur_encoder_out =
        GetEncoderOutFrame(model_->Allocator(), &encoder_out, t);
    cur_encoder_out =
        Repeat(model_->Allocator(), &cur_encoder_out, hyps_num_split);
    Ort::Value logit = model_->RunJoiner(
        std::move(cur_encoder_out), Clone(model_->Allocator(), &decoder_out));
    float *p_logit = logit.GetTensorMutableData<float>();

    for (int32_t b = 0; b < batch_size; ++b) {
      int32_t start = hyps_num_split[b];
      int32_t end = hyps_num_split[b + 1];
      LogSoftmax(p_logit, vocab_size, (end - start));
      auto topk =
          TopkIndex(p_logit, vocab_size * (end - start), max_active_paths_);

      Hypotheses hyps;
      for (auto i : topk) {
        int32_t hyp_index = i / vocab_size + start;
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
      p_logit += vocab_size * (end - start);
    }
  }

  for (int32_t b = 0; b != batch_size; ++b) {
    auto &hyps = cur[b];
    auto best_hyp = hyps.GetMostProbable(true);

    (*result)[b].hyps = std::move(hyps);
    (*result)[b].tokens = std::move(best_hyp.ys);
    (*result)[b].num_trailing_blanks = best_hyp.num_trailing_blanks;
  }
}

void OnlineTransducerModifiedBeamSearchDecoder::UpdateDecoderOut(
    OnlineTransducerDecoderResult *result) {
  if (result->tokens.size() == model_->ContextSize()) {
    result->decoder_out = Ort::Value{nullptr};
    return;
  }
  Ort::Value decoder_input = model_->BuildDecoderInput({*result});
  result->decoder_out = model_->RunDecoder(std::move(decoder_input));
}

}  // namespace sherpa_onnx
