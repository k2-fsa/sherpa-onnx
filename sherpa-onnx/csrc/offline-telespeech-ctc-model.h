// sherpa-onnx/csrc/offline-telespeech-ctc-model.h
//
// Copyright (c)  2024  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_OFFLINE_TELESPEECH_CTC_MODEL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_TELESPEECH_CTC_MODEL_H_
#include <memory>
#include <string>
#include <utility>
#include <vector>

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#include "onnxruntime_cxx_api.h"  // NOLINT
#include "sherpa-onnx/csrc/offline-ctc-model.h"
#include "sherpa-onnx/csrc/offline-model-config.h"

namespace sherpa_onnx {

/** This class implements the CTC model from
 * https://github.com/Tele-AI/TeleSpeech-ASR.
 *
 * See
 * https://github.com/lovemefan/telespeech-asr-python/blob/main/telespeechasr/onnx/onnx_infer.py
 * and
 * https://github.com/k2-fsa/sherpa-onnx/blob/master/scripts/tele-speech/test.py
 */
class OfflineTeleSpeechCtcModel : public OfflineCtcModel {
 public:
  explicit OfflineTeleSpeechCtcModel(const OfflineModelConfig &config);

#if __ANDROID_API__ >= 9
  OfflineTeleSpeechCtcModel(AAssetManager *mgr,
                            const OfflineModelConfig &config);
#endif

  ~OfflineTeleSpeechCtcModel() override;

  /** Run the forward method of the model.
   *
   * @param features  A tensor of shape (N, T, C).
   * @param features_length  A 1-D tensor of shape (N,) containing number of
   *                         valid frames in `features` before padding.
   *                         Its dtype is int64_t.
   *
   * @return Return a vector containing:
   *  - log_probs: A 3-D tensor of shape (N, T', vocab_size).
   *  - log_probs_length A 1-D tensor of shape (N,). Its dtype is int64_t
   */
  std::vector<Ort::Value> Forward(Ort::Value features,
                                  Ort::Value features_length) override;

  /** Return the vocabulary size of the model
   */
  int32_t VocabSize() const override;

  /** SubsamplingFactor of the model
   */
  int32_t SubsamplingFactor() const override;

  /** Return an allocator for allocating memory
   */
  OrtAllocator *Allocator() const override;

  // TeleSpeech CTC models do not support batch size > 1
  bool SupportBatchProcessing() const override { return false; }

  std::string FeatureNormalizationMethod() const override {
    return "per_feature";
  }

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_TELESPEECH_CTC_MODEL_H_
