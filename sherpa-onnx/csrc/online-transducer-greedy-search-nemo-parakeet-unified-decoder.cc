// sherpa-onnx/csrc/online-transducer-greedy-search-nemo-parakeet-unified-decoder.cc
//
// Copyright (c)  2026  Milan Leonard
// Copyright (c)  2026  Julian Pscheid

#include "sherpa-onnx/csrc/online-transducer-greedy-search-nemo-parakeet-unified-decoder.h"

#include <algorithm>
#include <array>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/math.h"
#include "sherpa-onnx/csrc/online-stream.h"
#include "sherpa-onnx/csrc/onnx-utils.h"

namespace sherpa_onnx {

static Ort::Value BuildDecoderInput(int32_t token, OrtAllocator *allocator) {
  std::array<int64_t, 2> shape{1, 1};
  Ort::Value decoder_input =
      Ort::Value::CreateTensor<int32_t>(allocator, shape.data(), shape.size());

  int32_t *p = decoder_input.GetTensorMutableData<int32_t>();
  p[0] = token;

  return decoder_input;
}

static void DecodeOne(const float *encoder_out, int32_t num_rows,
                      int32_t num_cols,
                      OnlineTransducerNeMoParakeetUnifiedModel *model,
                      float blank_penalty, OnlineStream *s) {
  auto memory_info =
      Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

  int32_t vocab_size = model->VocabSize();
  int32_t blank_id = vocab_size - 1;

  auto &r = s->GetResult();

  Ort::Value &last_decoder_out = s->GetNeMoDecoderOut();
  std::vector<Ort::Value> &last_decoder_states = s->GetNeMoDecoderStates();
  // Keep the predictor independent of segment results, which Reset() clears.
  // The cached state is after the token used to produce decoder_out.
  std::pair<Ort::Value, std::vector<Ort::Value>> decoder_output_pair{
      Ort::Value{nullptr}, std::vector<Ort::Value>{}};
  if (last_decoder_out) {
    decoder_output_pair.first = std::move(last_decoder_out);
    decoder_output_pair.second = std::move(last_decoder_states);
  } else {
    auto decoder_input = BuildDecoderInput(blank_id, model->Allocator());
    decoder_output_pair = model->RunDecoder(std::move(decoder_input),
                                            std::move(last_decoder_states));
  }

  std::array<int64_t, 3> encoder_shape{1, num_cols, 1};
  int32_t max_symbols_per_frame = 10;

  for (int32_t t = 0; t != num_rows; ++t) {
    Ort::Value cur_encoder_out = Ort::Value::CreateTensor(
        memory_info, const_cast<float *>(encoder_out) + t * num_cols, num_cols,
        encoder_shape.data(), encoder_shape.size());

    for (int32_t q = 0; q != max_symbols_per_frame; ++q) {
      Ort::Value logit = model->RunJoiner(View(&cur_encoder_out),
                                          View(&decoder_output_pair.first));

      float *p_logit = logit.GetTensorMutableData<float>();
      if (blank_penalty > 0) {
        p_logit[blank_id] -= blank_penalty;
      }

      int32_t y = MaxElementIndex(p_logit, vocab_size);

      // Apply LogSoftmax and get log probability for selected token
      LogSoftmax(p_logit, vocab_size);
      float log_prob = p_logit[y];

      if (y != blank_id) {
        r.tokens.push_back(y);
        r.timestamps.push_back(t + r.frame_offset);
        r.ys_probs.push_back(log_prob);
        r.num_trailing_blanks = 0;

        auto decoder_input = BuildDecoderInput(y, model->Allocator());
        decoder_output_pair = model->RunDecoder(
            std::move(decoder_input), std::move(decoder_output_pair.second));
      } else {
        ++r.num_trailing_blanks;
        break;
      }
    }
  }

  s->SetNeMoDecoderOut(std::move(decoder_output_pair.first));
  s->SetNeMoDecoderStates(std::move(decoder_output_pair.second));

  r.frame_offset += num_rows;
}

void OnlineTransducerGreedySearchNeMoParakeetUnifiedDecoder::Decode(
    Ort::Value encoder_out, OnlineStream **ss, int32_t n) const {
  auto shape = encoder_out.GetTensorTypeAndShapeInfo().GetShape();
  int32_t batch_size = static_cast<int32_t>(shape[0]);

  if (batch_size != n) {
    SHERPA_ONNX_LOGE("Size mismatch! encoder_out.size(0) %d, n: %d",
                     static_cast<int32_t>(shape[0]), n);
    SHERPA_ONNX_EXIT(-1);
  }

  int32_t dim1 = static_cast<int32_t>(shape[1]);
  int32_t dim2 = static_cast<int32_t>(shape[2]);
  const float *p = encoder_out.GetTensorData<float>();

  for (int32_t i = 0; i != batch_size; ++i) {
    const float *this_p = p + dim1 * dim2 * i;
    DecodeOne(this_p, dim1, dim2, model_, blank_penalty_, ss[i]);
  }
}

}  // namespace sherpa_onnx
