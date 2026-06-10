// sherpa-onnx/csrc/offline-diacritization-catt-impl.h
//
// Copyright (c)  2026  Matias Lin
#ifndef SHERPA_ONNX_CSRC_OFFLINE_DIACRITIZATION_CATT_IMPL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_DIACRITIZATION_CATT_IMPL_H_

#include <algorithm>
#include <array>
#include <cstdint>
#include <iterator>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/offline-catt-model.h"
#include "sherpa-onnx/csrc/offline-diacritization-impl.h"
#include "sherpa-onnx/csrc/tashkeel-tokenizer.h"

namespace sherpa_onnx {

class OfflineDiacritizationCATTImpl : public OfflineDiacritizationImpl {
 public:
  explicit OfflineDiacritizationCATTImpl(
      const OfflineDiacritizationConfig &config)
      : config_(config), model_(config.model) {}

  template <typename Manager>
  OfflineDiacritizationCATTImpl(Manager *mgr,
                                const OfflineDiacritizationConfig &config)
      : config_(config), model_(mgr, config.model) {}

  std::string AddDiacritics(const std::string &text) const override {
    if (text.empty()) {
      return {};
    }

    // Tokenizing inputs
    auto enc = tokenizer_.Encode(text);
    const auto &input_ids = enc.input_ids_;
    if (input_ids.size() <= 2) {
      // Nothing between BOS and EOS (i.e. no letters to diacritize)
      return text;
    }

    // Preparing encode inputs: src and src_mask
    const int64_t src_seq_len = static_cast<int64_t>(input_ids.size()) - 2;
    std::vector<int64_t> stripped(input_ids.begin() + 1, input_ids.end() - 1);

    auto memory_info =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

    std::array<int64_t, 2> src_shape = {1, src_seq_len};
    Ort::Value src = Ort::Value::CreateTensor<int64_t>(
        memory_info, stripped.data(), stripped.size(), src_shape.data(),
        src_shape.size());

    std::array<int64_t, 4> src_mask_shape = {1, 1, src_seq_len, src_seq_len};
    Ort::Value src_mask = Ort::Value::CreateTensor<bool>(
        model_.Allocator(), src_mask_shape.data(), src_mask_shape.size());
    bool *src_mask_data = src_mask.GetTensorMutableData<bool>();
    std::fill(src_mask_data, src_mask_data + src_seq_len * src_seq_len, true);

    // Encoder + Decoder runs
    Ort::Value enc_src = model_.RunEncoder(std::move(src), std::move(src_mask));
    Ort::Value logits = model_.RunDecoder(std::move(enc_src));

    // Apply argmax
    auto logits_shape = logits.GetTensorTypeAndShapeInfo().GetShape();
    if (logits_shape.size() != 3 || logits_shape[0] != 1 ||
        logits_shape[1] != src_seq_len || logits_shape[2] <= 0) {
      SHERPA_ONNX_LOGE(
          "Logits is of incorrect shape. Expected [1, %d, "
          "tashkeel_vocab_size].",
          static_cast<int>(src_seq_len));
      SHERPA_ONNX_EXIT(-1);
    }
    const int64_t tashkeel_vocab_size = logits_shape[2];
    const float *probs = logits.GetTensorData<float>();
    std::vector<int64_t> pred_ids(static_cast<size_t>(src_seq_len));
    for (int64_t i = 0; i < src_seq_len; ++i, probs += tashkeel_vocab_size) {
      pred_ids[i] = static_cast<int64_t>(std::distance(
          probs, std::max_element(probs, probs + tashkeel_vocab_size)));
    }

    // Apply space mask
    const int64_t space_id = tokenizer_.SpaceLetterId();
    const int64_t nt_id = tokenizer_.NoTashkeelId();
    for (int64_t i = 0; i < src_seq_len; ++i) {
      if (stripped[i] == space_id) {
        pred_ids[i] = nt_id;
      }
    }

    return tokenizer_.Decode(input_ids, pred_ids);
  }

 private:
  OfflineDiacritizationConfig config_;
  TashkeelTokenizer tokenizer_;
  OfflineCATTModel model_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_DIACRITIZATION_CATT_IMPL_H_
