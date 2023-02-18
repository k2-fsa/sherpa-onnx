// sherpa/csrc/decode.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/decode.h"

#include <assert.h>

#include <algorithm>
#include <utility>
#include <vector>

namespace sherpa_onnx {

static Ort::Value Clone(Ort::Value *v) {
  auto type_and_shape = v->GetTensorTypeAndShapeInfo();
  std::vector<int64_t> shape = type_and_shape.GetShape();

  auto memory_info =
      Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

  return Ort::Value::CreateTensor(memory_info, v->GetTensorMutableData<float>(),
                                  type_and_shape.GetElementCount(),
                                  shape.data(), shape.size());
}

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

void GreedySearch(OnlineTransducerModel *model, Ort::Value encoder_out,
                  std::vector<int64_t> *hyp) {
  std::vector<int64_t> encoder_out_shape =
      encoder_out.GetTensorTypeAndShapeInfo().GetShape();

  if (encoder_out_shape[0] > 1) {
    fprintf(stderr, "Only batch_size=1 is implemented. Given: %d\n",
            static_cast<int32_t>(encoder_out_shape[0]));
  }

  int32_t num_frames = encoder_out_shape[1];
  int32_t vocab_size = model->VocabSize();

  Ort::Value decoder_input = model->BuildDecoderInput(*hyp);
  Ort::Value decoder_out = model->RunDecoder(std::move(decoder_input));

  for (int32_t t = 0; t != num_frames; ++t) {
    Ort::Value cur_encoder_out = GetFrame(&encoder_out, t);
    Ort::Value logit =
        model->RunJoiner(std::move(cur_encoder_out), Clone(&decoder_out));
    const float *p_logit = logit.GetTensorData<float>();

    auto y = static_cast<int32_t>(std::distance(
        static_cast<const float *>(p_logit),
        std::max_element(static_cast<const float *>(p_logit),
                         static_cast<const float *>(p_logit) + vocab_size)));
    if (y != 0) {
      hyp->push_back(y);
      decoder_input = model->BuildDecoderInput(*hyp);
      decoder_out = model->RunDecoder(std::move(decoder_input));
    }
  }
}

}  // namespace sherpa_onnx
