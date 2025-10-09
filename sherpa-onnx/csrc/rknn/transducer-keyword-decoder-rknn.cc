// sherpa-onnx/csrc/rknn/transducer-keywords-decoder-rknn.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa-onnx/csrc/rknn/transducer-keyword-decoder-rknn.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/log.h"

namespace sherpa_onnx {

TransducerKeywordResult TransducerKeywordDecoderRknn::GetEmptyResult() const {
  int32_t context_size = model_->ContextSize();
  int32_t blank_id = 0;  // always 0
  TransducerKeywordResult r;
  std::vector<int64_t> blanks(context_size, -1);
  blanks.back() = blank_id;

  Hypotheses blank_hyp({{blanks, 0}});
  r.hyps = std::move(blank_hyp);
  return r;
}

std::vector<std::vector<float>> GetDecoderOut(
    OnlineZipformerTransducerModelRknn *model, const Hypotheses &hyp_vec);

std::vector<std::vector<float>> GetJoinerOutLogSoftmax(
    OnlineZipformerTransducerModelRknn *model, const float *p_encoder_out,
    const std::vector<std::vector<float>> &decoder_out);

void TransducerKeywordDecoderRknn::Decode(std::vector<float> encoder_out,
                                          OnlineStreamRknn *s) {
  auto attr = model_->GetEncoderOutAttr();
  int32_t num_frames = attr.dims[1];
  int32_t encoder_out_dim = attr.dims[2];

  int32_t vocab_size = model_->VocabSize();
  int32_t context_size = model_->ContextSize();

  std::vector<int64_t> blanks(context_size, -1);
  blanks.back() = 0;  // blank_id is hardcoded to 0

  auto r = s->GetKeywordResult();

  Hypotheses cur = std::move(r.hyps);
  std::vector<Hypothesis> prev;

  auto decoder_out = GetDecoderOut(model_, cur);

  const float *p_encoder_out = encoder_out.data();

  int32_t frame_offset = r.frame_offset;

  for (int32_t t = 0; t != num_frames; ++t) {
    prev = cur.Vec();
    cur.Clear();

    auto log_probs = GetJoinerOutLogSoftmax(model_, p_encoder_out, decoder_out);

    auto log_probs_old = log_probs;

    p_encoder_out += encoder_out_dim;

    for (int32_t i = 0; i != prev.size(); ++i) {
      auto log_prob = prev[i].log_prob;
      for (auto &p : log_probs[i]) {
        p += log_prob;
      }
    }

    auto topk = TopkIndex(log_probs, max_active_paths_);

    Hypotheses hyps;

    for (auto k : topk) {
      int32_t hyp_index = k / vocab_size;
      int32_t new_token = k % vocab_size;

      Hypothesis new_hyp = prev[hyp_index];
      float context_score = 0;
      auto context_state = new_hyp.context_state;

      // blank is hardcoded to 0
      // also, it treats unk as blank
      if (new_token != 0 && new_token != unk_id_) {
        new_hyp.ys.push_back(new_token);
        new_hyp.timestamps.push_back(t + frame_offset);
        new_hyp.ys_probs.push_back(exp(log_probs_old[hyp_index][new_token]));

        new_hyp.num_trailing_blanks = 0;
        auto context_res =
            s->GetContextGraph()->ForwardOneStep(context_state, new_token);
        context_score = std::get<0>(context_res);
        new_hyp.context_state = std::get<1>(context_res);
        // Start matching from the start state, forget the decoder history.
        if (new_hyp.context_state->token == -1) {
          new_hyp.ys = blanks;
          new_hyp.timestamps.clear();
          new_hyp.ys_probs.clear();
        }
      } else {
        ++new_hyp.num_trailing_blanks;
      }
      new_hyp.log_prob = log_probs[hyp_index][new_token] + context_score;
      hyps.Add(std::move(new_hyp));
    }  // for (auto k : topk)

    auto best_hyp = hyps.GetMostProbable(false);

    auto status = s->GetContextGraph()->IsMatched(best_hyp.context_state);
    bool matched = std::get<0>(status);
    const ContextState *matched_state = std::get<1>(status);

    if (matched) {
      float ys_prob = 0.0;
      for (int32_t i = 0; i < matched_state->level; ++i) {
        ys_prob += best_hyp.ys_probs[i];
      }
      ys_prob /= matched_state->level;
      if (best_hyp.num_trailing_blanks > num_trailing_blanks_ &&
          ys_prob >= matched_state->ac_threshold) {
        r.tokens = {best_hyp.ys.end() - matched_state->level,
                    best_hyp.ys.end()};
        r.timestamps = {best_hyp.timestamps.end() - matched_state->level,
                        best_hyp.timestamps.end()};
        r.keyword = matched_state->phrase;

        hyps = Hypotheses({{blanks, 0, s->GetContextGraph()->Root()}});
      }
    }

    cur = std::move(hyps);
    decoder_out = GetDecoderOut(model_, cur);
  }

  auto best_hyp = cur.GetMostProbable(false);
  r.hyps = std::move(cur);
  r.frame_offset += num_frames;
  r.num_trailing_blanks = best_hyp.num_trailing_blanks;

  s->SetKeywordResult(r);
}

}  // namespace sherpa_onnx
