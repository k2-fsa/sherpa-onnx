// sherpa-onnx/csrc/offline-transducer-greedy-search-nemo-decoder.cc
//
// Copyright (c)  2024  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-transducer-greedy-search-nemo-decoder.h"

#include <algorithm>
#include <iterator>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/math.h"
#include "sherpa-onnx/csrc/onnx-utils.h"

namespace sherpa_onnx {

static std::pair<Ort::Value, Ort::Value> BuildDecoderInput(
    int32_t token, OrtAllocator *allocator) {
  std::array<int64_t, 2> shape{1, 1};

  Ort::Value decoder_input =
      Ort::Value::CreateTensor<int32_t>(allocator, shape.data(), shape.size());

  std::array<int64_t, 1> length_shape{1};
  Ort::Value decoder_input_length = Ort::Value::CreateTensor<int32_t>(
      allocator, length_shape.data(), length_shape.size());

  int32_t *p = decoder_input.GetTensorMutableData<int32_t>();

  int32_t *p_length = decoder_input_length.GetTensorMutableData<int32_t>();

  p[0] = token;

  p_length[0] = 1;

  return {std::move(decoder_input), std::move(decoder_input_length)};
}

static OfflineTransducerDecoderResult DecodeOne(
    const float *p, int32_t num_rows, int32_t num_cols,
    OfflineTransducerNeMoModel *model, float blank_penalty) {
  auto memory_info =
      Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

  OfflineTransducerDecoderResult ans;

  int32_t vocab_size = model->VocabSize();
  int32_t blank_id = vocab_size - 1;
  int32_t max_symbols_per_frame = 10;

  auto decoder_input_pair = BuildDecoderInput(blank_id, model->Allocator());

  std::pair<Ort::Value, std::vector<Ort::Value>> decoder_output_pair =
      model->RunDecoder(std::move(decoder_input_pair.first),
                        std::move(decoder_input_pair.second),
                        model->GetDecoderInitStates(1));

  std::array<int64_t, 3> encoder_shape{1, num_cols, 1};

  for (int32_t t = 0; t != num_rows; ++t) {
    Ort::Value cur_encoder_out = Ort::Value::CreateTensor(
        memory_info, const_cast<float *>(p) + t * num_cols, num_cols,
        encoder_shape.data(), encoder_shape.size());

    for (int32_t q = 0; q != max_symbols_per_frame; ++q) {
      Ort::Value logit = model->RunJoiner(View(&cur_encoder_out),
                                          View(&decoder_output_pair.first));

      float *p_logit = logit.GetTensorMutableData<float>();
      if (blank_penalty > 0) {
        p_logit[blank_id] -= blank_penalty;
      }

      auto y = static_cast<int32_t>(std::distance(
          static_cast<const float *>(p_logit),
          std::max_element(static_cast<const float *>(p_logit),
                           static_cast<const float *>(p_logit) + vocab_size)));

      // Apply LogSoftmax and get log probability for selected token
      LogSoftmax(p_logit, vocab_size);
      float log_prob = p_logit[y];

      if (y != blank_id) {
        ans.tokens.push_back(y);
        ans.timestamps.push_back(t);
        ans.ys_log_probs.push_back(log_prob);

        decoder_input_pair = BuildDecoderInput(y, model->Allocator());

        decoder_output_pair =
            model->RunDecoder(std::move(decoder_input_pair.first),
                              std::move(decoder_input_pair.second),
                              std::move(decoder_output_pair.second));
      } else {
        break;
      }  // if (y != blank_id)
    }
  }  // for (int32_t i = 0; i != num_rows; ++i)

  return ans;
}

static OfflineTransducerDecoderResult DecodeOneTDT(
    const float *p, int32_t num_rows, int32_t num_cols,
    OfflineTransducerNeMoModel *model, float blank_penalty) {
  auto memory_info =
      Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

  OfflineTransducerDecoderResult ans;

  int32_t vocab_size = model->VocabSize();
  int32_t blank_id = vocab_size - 1;

  auto decoder_input_pair = BuildDecoderInput(blank_id, model->Allocator());

  std::pair<Ort::Value, std::vector<Ort::Value>> decoder_output_pair =
      model->RunDecoder(std::move(decoder_input_pair.first),
                        std::move(decoder_input_pair.second),
                        model->GetDecoderInitStates(1));

  std::array<int64_t, 3> encoder_shape{1, num_cols, 1};

  int32_t max_tokens_per_frame = 5;
  int32_t tokens_this_frame = 0;

  int32_t skip = 0;
  for (int32_t t = 0; t < num_rows; t += skip) {
    Ort::Value cur_encoder_out = Ort::Value::CreateTensor(
        memory_info, const_cast<float *>(p) + t * num_cols, num_cols,
        encoder_shape.data(), encoder_shape.size());

    Ort::Value logit = model->RunJoiner(View(&cur_encoder_out),
                                        View(&decoder_output_pair.first));

    auto shape = logit.GetTensorTypeAndShapeInfo().GetShape();

    float *p_logit = logit.GetTensorMutableData<float>();
    if (blank_penalty > 0) {
      p_logit[blank_id] -= blank_penalty;
    }

    int32_t output_size = shape.back();
    int32_t num_durations = output_size - vocab_size;

    // Split logits into token and duration logits
    const float *token_logits = p_logit;
    const float *duration_logits = p_logit + vocab_size;

    auto y = static_cast<int32_t>(std::distance(
        token_logits,
        std::max_element(token_logits, token_logits + vocab_size)));

    // Apply LogSoftmax to token logits and get log probability
    // Note: Need to make a copy since token_logits is const
    std::vector<float> token_logits_copy(token_logits, token_logits + vocab_size);
    LogSoftmax(token_logits_copy.data(), vocab_size);
    float log_prob = token_logits_copy[y];

    // note that skip can be 0
    skip = static_cast<int32_t>(std::distance(
        duration_logits,
        std::max_element(duration_logits, duration_logits + num_durations)));

    if (y != blank_id) {
      ans.tokens.push_back(y);
      ans.timestamps.push_back(t);
      ans.durations.push_back(skip);
      ans.ys_log_probs.push_back(log_prob);

      decoder_input_pair = BuildDecoderInput(y, model->Allocator());

      decoder_output_pair =
          model->RunDecoder(std::move(decoder_input_pair.first),
                            std::move(decoder_input_pair.second),
                            std::move(decoder_output_pair.second));

      tokens_this_frame += 1;
    }

    if (skip > 0) {
      tokens_this_frame = 0;
    }

    if (tokens_this_frame >= max_tokens_per_frame) {
      tokens_this_frame = 0;
      skip = 1;
    }

    if (y == blank_id && skip == 0) {
      tokens_this_frame = 0;
      skip = 1;
    }
  }  // for (int32_t t = 0; t < num_rows; t += skip)

  return ans;
}

std::vector<OfflineTransducerDecoderResult>
OfflineTransducerGreedySearchNeMoDecoder::Decode(
    Ort::Value encoder_out, Ort::Value encoder_out_length,
    OfflineStream ** /*ss = nullptr*/, int32_t /*n= 0*/) {
  auto shape = encoder_out.GetTensorTypeAndShapeInfo().GetShape();

  int32_t batch_size = static_cast<int32_t>(shape[0]);
  int32_t dim1 = static_cast<int32_t>(shape[1]);
  int32_t dim2 = static_cast<int32_t>(shape[2]);

  auto length_type =
      encoder_out_length.GetTensorTypeAndShapeInfo().GetElementType();
  if ((length_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) &&
      (length_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64)) {
    SHERPA_ONNX_LOGE("Unsupported encoder_out_length data type: %d",
                     static_cast<int32_t>(length_type));
    SHERPA_ONNX_EXIT(-1);
  }

  const float *p = encoder_out.GetTensorData<float>();

  std::vector<OfflineTransducerDecoderResult> ans(batch_size);

  for (int32_t i = 0; i != batch_size; ++i) {
    const float *this_p = p + dim1 * dim2 * i;
    int32_t this_len = length_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32
                           ? encoder_out_length.GetTensorData<int32_t>()[i]
                           : encoder_out_length.GetTensorData<int64_t>()[i];

    if (is_tdt_) {
      ans[i] = DecodeOneTDT(this_p, this_len, dim2, model_, blank_penalty_);
    } else {
      ans[i] = DecodeOne(this_p, this_len, dim2, model_, blank_penalty_);
    }
  }

  return ans;
}

}  // namespace sherpa_onnx
