// sherpa-onnx/csrc/online-cnn-bilstm-model-meta-data.h
//
// Copyright (c) 2024 Jian You (jianyou@cisco.com, Cisco Systems)

#ifndef SHERPA_ONNX_CSRC_ONLINE_CNN_BILSTM_MODEL_META_DATA_H_
#define SHERPA_ONNX_CSRC_ONLINE_CNN_BILSTM_MODEL_META_DATA_H_

namespace sherpa_onnx {

struct OnlineCNNBiLSTMModelMetaData {
  int32_t comma_id;
  int32_t period_id;
  int32_t quest_id;

  int32_t upper_id;
  int32_t cap_id;
  int32_t mix_case_id;

  int32_t num_cases;
  int32_t num_punctuations;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_ONLINE_CNN_BILSTM_MODEL_META_DATA_H_
