// sherpa-onnx/csrc/rknn/online-transducer-modified-beam-search-decoder-rknn.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa-onnx/csrc/rknn/online-transducer-modified-beam-search-decoder-rknn.h"

#include <algorithm>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/hypothesis.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/math.h"

namespace sherpa_onnx {

OnlineTransducerDecoderResultRknn
OnlineTransducerModifiedBeamSearchDecoderRknn::GetEmptyResult() const {
  int32_t context_size = model_->ContextSize();
  int32_t blank_id = 0;  // always 0
  OnlineTransducerDecoderResultRknn r;

  std::vector<int64_t> blanks(context_size, -1);
  blanks.back() = blank_id;

  Hypotheses blank_hyp({{blanks, 0}});
  r.hyps = std::move(blank_hyp);
  r.tokens = std::move(blanks);

  return r;
}

void OnlineTransducerModifiedBeamSearchDecoderRknn::StripLeadingBlanks(
    OnlineTransducerDecoderResultRknn *r) const {
  int32_t context_size = model_->ContextSize();
  auto hyp = r->hyps.GetMostProbable(true);

  std::vector<int64_t> tokens(hyp.ys.begin() + context_size, hyp.ys.end());
  r->tokens = std::move(tokens);
  r->timestamps = std::move(hyp.timestamps);

  r->num_trailing_blanks = hyp.num_trailing_blanks;
}

static std::vector<std::vector<float>> GetDecoderOut(
    OnlineZipformerTransducerModelRknn *model, const Hypotheses &hyp_vec) {
  std::vector<std::vector<float>> ans;
  ans.reserve(hyp_vec.Size());

  int32_t context_size = model->ContextSize();
  for (const auto &p : hyp_vec) {
    const auto &hyp = p.second;
    auto start = hyp.ys.begin() + (hyp.ys.size() - context_size);
    auto end = hyp.ys.end();
    auto tokens = std::vector<int64_t>(start, end);
    auto decoder_out = model->RunDecoder(std::move(tokens));

    ans.push_back(std::move(decoder_out));
  }

  return ans;
}

static std::vector<std::vector<float>> GetJoinerOutLogSoftmax(
    OnlineZipformerTransducerModelRknn *model, const float *p_encoder_out,
    const std::vector<std::vector<float>> &decoder_out) {
  std::vector<std::vector<float>> ans;
  ans.reserve(decoder_out.size());

  for (const auto &d : decoder_out) {
    auto joiner_out = model->RunJoiner(p_encoder_out, d.data());

    LogSoftmax(joiner_out.data(), joiner_out.size());

    ans.push_back(std::move(joiner_out));
  }
  return ans;
}

void OnlineTransducerModifiedBeamSearchDecoderRknn::Decode(
    std::vector<float> encoder_out,
    OnlineTransducerDecoderResultRknn *result) const {
  auto &r = result[0];
  auto attr = model_->GetEncoderOutAttr();
  int32_t num_frames = attr.dims[1];
  int32_t encoder_out_dim = attr.dims[2];

  int32_t vocab_size = model_->VocabSize();
  int32_t context_size = model_->ContextSize();

  Hypotheses cur = std::move(result->hyps);
  std::vector<Hypothesis> prev;

  auto decoder_out = std::move(result->previous_decoder_out2);
  if (decoder_out.empty()) {
    decoder_out = GetDecoderOut(model_, cur);
  }

  const float *p_encoder_out = encoder_out.data();

  int32_t frame_offset = result->frame_offset;

  for (int32_t t = 0; t != num_frames; ++t) {
    prev = cur.Vec();
    cur.Clear();

    auto log_probs = GetJoinerOutLogSoftmax(model_, p_encoder_out, decoder_out);
    p_encoder_out += encoder_out_dim;

    for (int32_t i = 0; i != prev.size(); ++i) {
      auto log_prob = prev[i].log_prob;
      for (auto &p : log_probs[i]) {
        p += log_prob;
      }
    }

    auto topk = TopkIndex(log_probs, max_active_paths_);
    for (auto k : topk) {
      int32_t hyp_index = k / vocab_size;
      int32_t new_token = k % vocab_size;

      Hypothesis new_hyp = prev[hyp_index];
      new_hyp.log_prob = log_probs[hyp_index][new_token];

      // blank is hardcoded to 0
      // also, it treats unk as blank
      if (new_token != 0 && new_token != unk_id_) {
        new_hyp.ys.push_back(new_token);
        new_hyp.timestamps.push_back(t + frame_offset);
        new_hyp.num_trailing_blanks = 0;

      } else {
        ++new_hyp.num_trailing_blanks;
      }
      cur.Add(std::move(new_hyp));
    }

    decoder_out = GetDecoderOut(model_, cur);
  }

  result->hyps = std::move(cur);
  result->frame_offset += num_frames;
  result->previous_decoder_out2 = std::move(decoder_out);
}

}  // namespace sherpa_onnx
