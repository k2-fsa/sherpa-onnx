// sherpa-onnx/csrc/offline-w2v-bert-ctc-model-config.h
//
// Copyright (c)  2026  graehl
#ifndef SHERPA_ONNX_CSRC_OFFLINE_W2V_BERT_CTC_MODEL_CONFIG_H_
#define SHERPA_ONNX_CSRC_OFFLINE_W2V_BERT_CTC_MODEL_CONFIG_H_

#include <string>

#include "sherpa-onnx/csrc/parse-options.h"

namespace sherpa_onnx {

// A w2v-BERT 2.0 CTC model exported by scripts/w2v-bert/export_onnx.py.
struct OfflineW2vBertCtcModelConfig {
  std::string model;

  OfflineW2vBertCtcModelConfig() = default;
  explicit OfflineW2vBertCtcModelConfig(const std::string &model)
      : model(model) {}

  void Register(ParseOptions *po);
  bool Validate() const;

  std::string ToString() const;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_W2V_BERT_CTC_MODEL_CONFIG_H_
