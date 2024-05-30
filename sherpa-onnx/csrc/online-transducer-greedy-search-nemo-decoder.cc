// sherpa-onnx/csrc/online-transducer-greedy-search-nemo-decoder.cc
//
// Copyright (c)  2024  Xiaomi Corporation
// Copyright (c)  2024  Sangeet Sagar

#include "sherpa-onnx/csrc/online-transducer-greedy-search-nemo-decoder.h"

#include <algorithm>
#include <iterator>
#include <utility>

#include "sherpa-onnx/csrc/macros.h"
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


OnlineTransducerDecoderResult
OnlineTransducerGreedySearchNeMoDecoder::GetEmptyResult() const {
  int32_t context_size = 8;
  int32_t blank_id = 0;  // always 0
  OnlineTransducerDecoderResult r;
  r.tokens.resize(context_size, -1);
  r.tokens.back() = blank_id;

  return r;
}

static void UpdateCachedDecoderOut(
    OrtAllocator *allocator, const Ort::Value *decoder_out,
    std::vector<OnlineTransducerDecoderResult> *result) {
  std::vector<int64_t> shape =
      decoder_out->GetTensorTypeAndShapeInfo().GetShape();
  auto memory_info =
      Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
  std::array<int64_t, 2> v_shape{1, shape[1]};

  const float *src = decoder_out->GetTensorData<float>();
  for (auto &r : *result) {
    if (!r.decoder_out) {
      r.decoder_out = Ort::Value::CreateTensor<float>(allocator, v_shape.data(),
                                                      v_shape.size());
    }

    float *dst = r.decoder_out.GetTensorMutableData<float>();
    std::copy(src, src + shape[1], dst);
    src += shape[1];
  }
}

std::vector<Ort::Value> DecodeOne(
    const float *encoder_out, int32_t num_rows, int32_t num_cols,
    OnlineTransducerNeMoModel *model, float blank_penalty,
    std::vector<Ort::Value>& decoder_states,
    std::vector<OnlineTransducerDecoderResult> *result) {

  auto memory_info =
      Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

  // OnlineTransducerDecoderResult result;
  int32_t vocab_size = model->VocabSize();
  int32_t blank_id = vocab_size - 1;
  
  auto &r = (*result)[0];
  Ort::Value decoder_out{nullptr};

  auto decoder_input_pair = BuildDecoderInput(blank_id, model->Allocator());
  // decoder_input_pair[0]: decoder_input
  // decoder_input_pair[1]: decoder_input_length (discarded)

  // decoder_output_pair.second returns the next decoder state
  std::pair<Ort::Value, std::vector<Ort::Value>> decoder_output_pair =
      model->RunDecoder(std::move(decoder_input_pair.first),
                         std::move(decoder_states)); // here decoder_states = {len=0, cap=0}. But decoder_output_pair= {first, second: {len=2, cap=2}} // ATTN

  std::array<int64_t, 3> encoder_shape{1, num_cols, 1};

  decoder_states = std::move(decoder_output_pair.second);

  // TODO: Inside this loop, I need to framewise decoding.
  for (int32_t t = 0; t != num_rows; ++t) {
    Ort::Value cur_encoder_out = Ort::Value::CreateTensor(
        memory_info, const_cast<float *>(encoder_out) + t * num_cols, num_cols,
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
    SHERPA_ONNX_LOGE("y=%d", y);
    if (y != blank_id) {
      r.tokens.push_back(y);
      r.timestamps.push_back(t + r.frame_offset);

      decoder_input_pair = BuildDecoderInput(y, model->Allocator());

      // last decoder state becomes the current state for the first chunk
      decoder_output_pair =
          model->RunDecoder(std::move(decoder_input_pair.first),
                             std::move(decoder_states));

      // Update the decoder states for the next chunk
      decoder_states = std::move(decoder_output_pair.second);
    }
  }

  decoder_out = std::move(decoder_output_pair.first);
//  UpdateCachedDecoderOut(model->Allocator(), &decoder_out, result);

  // Update frame_offset
  for (auto &r : *result) {
    r.frame_offset += num_rows;
  }

  return std::move(decoder_states);
}


std::vector<Ort::Value> OnlineTransducerGreedySearchNeMoDecoder::Decode(
    Ort::Value encoder_out, 
    std::vector<Ort::Value> decoder_states,
    std::vector<OnlineTransducerDecoderResult> *result,
    OnlineStream ** /*ss = nullptr*/, int32_t /*n= 0*/) {

  auto shape = encoder_out.GetTensorTypeAndShapeInfo().GetShape();

  if (shape[0] != result->size()) {
    SHERPA_ONNX_LOGE(
        "Size mismatch! encoder_out.size(0) %d, result.size(0): %d",
        static_cast<int32_t>(shape[0]),
        static_cast<int32_t>(result->size()));
    exit(-1);
  }

  int32_t batch_size = static_cast<int32_t>(shape[0]);  // bs = 1
  int32_t dim1 = static_cast<int32_t>(shape[1]); // 2
  int32_t dim2 = static_cast<int32_t>(shape[2]); // 512

  // Define and initialize encoder_out_length
  Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
  
  int64_t length_value = 1;
  std::vector<int64_t> length_shape = {1};
  
  Ort::Value encoder_out_length = Ort::Value::CreateTensor<int64_t>(
      memory_info, &length_value, 1, length_shape.data(), length_shape.size()
  );

  const int64_t *p_length = encoder_out_length.GetTensorData<int64_t>();
  const float *p = encoder_out.GetTensorData<float>();

  // std::vector<OnlineTransducerDecoderResult> ans(batch_size);

  for (int32_t i = 0; i != batch_size; ++i) {
    const float *this_p = p + dim1 * dim2 * i;
    int32_t this_len = p_length[i];

    // outputs the decoder state from last chunk.
    auto last_decoder_states = DecodeOne(this_p, this_len, dim2, model_, blank_penalty_, decoder_states, result);
    // ans[i] = decode_result_pair.first;
    decoder_states = std::move(last_decoder_states);
  }
  
  return decoder_states;

}

} // namespace sherpa_onnx