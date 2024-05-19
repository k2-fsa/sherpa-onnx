// sherpa-onnx/csrc/online-transducer-greedy-search-nemo-decoder.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/online-transducer-greedy-search-nemo-decoder.h"

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

static OnlineTransducerDecoderResult DecodeOne(
    const float *p, int32_t num_rows, int32_t num_cols,
    OnlineTransducerNeMoModel *model, float blank_penalty) {
  auto memory_info =
      Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

  OnlineTransducerDecoderResult ans;

  int32_t vocab_size = model->VocabSize();
  int32_t blank_id = vocab_size - 1;

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

    Ort::Value logit = model->RunJoiner(std::move(cur_encoder_out),
                                        View(&decoder_output_pair.first));

    float *p_logit = logit.GetTensorMutableData<float>();
    if (blank_penalty > 0) {
      
      p_logit[blank_id] -= blank_penalty;
    }

    auto y = static_cast<int32_t>(std::distance(
        static_cast<const float *>(p_logit),
        std::max_element(static_cast<const float *>(p_logit),
                         static_cast<const float *>(p_logit) + vocab_size)));

    if (y != blank_id) {
      ans.tokens.push_back(y);
      ans.timestamps.push_back(t);

      decoder_input_pair = BuildDecoderInput(y, model->Allocator());

      decoder_output_pair =
          model->RunDecoder(std::move(decoder_input_pair.first),
                            std::move(decoder_input_pair.second),
                            std::move(decoder_output_pair.second));
    }  // if (y != blank_id)
  }    // for (int32_t i = 0; i != num_rows; ++i)

  return ans;
}

std::vector<OnlineTransducerDecoderResult>
OnlineTransducerGreedySearchNeMoDecoder::Decode(
    Ort::Value encoder_out, 
    std::vector<OnlineTransducerDecoderResult> *result) {
  auto shape = encoder_out.GetTensorTypeAndShapeInfo().GetShape();

  int32_t batch_size = static_cast<int32_t>(shape[0]);
  int32_t dim1 = static_cast<int32_t>(shape[1]);
  int32_t dim2 = static_cast<int32_t>(shape[2]);

  const float *p = encoder_out.GetTensorData<float>();
  
  // checking for non-null elements in results

  // create a new tensor with modified shape based on 
  // the first element of result and use cached decoder_out 
  // values if available.

  // For each frame (num of frames is given by dim2), compute logits, 
  // determine tokens, and update results, 
  // then regenerate decoder output
  // if tokens are emitted.

  // call UpdateCachedDecoderOut and update frame offset
}

}  // namespace sherpa_onnx