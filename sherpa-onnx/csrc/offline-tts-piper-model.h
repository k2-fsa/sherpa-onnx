// sherpa-onnx/csrc/offline-tts-piper-model.h
//
// Copyright (c)  2025  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_OFFLINE_TTS_PIPER_MODEL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_TTS_PIPER_MODEL_H_

#include <memory>
#include <string>
#include <vector>

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#include "onnxruntime_cxx_api.h"
#include "sherpa-onnx/csrc/offline-tts-model-config.h"
#include "sherpa-onnx/csrc/offline-tts-piper-model-meta-data.h"

namespace sherpa_onnx {

class OfflineTtsPiperModel {
 public:
  explicit OfflineTtsPiperModel(const OfflineTtsModelConfig &config);

#if __ANDROID_API__ >= 9
  OfflineTtsPiperModel(AAssetManager *mgr,
                       const OfflineTtsModelConfig &config);
#endif

  ~OfflineTtsPiperModel();

  /** Run the model
   *
   * @param phoneme_ids Token IDs of the text. Its shape is (num_phonemes,)
   * @param speaker_id  Speaker ID. If the model supports only a single
   *                    speaker, then it should be 0. 
   * @param speed Speech speed. Defaults to 1.0. If it is larger, speech is 
   *              faster; if it is smaller, speech is slower.
   * @return Return a tensor of shape (1, num_samples)
   */
  Ort::Value Run(Ort::Value phoneme_ids, int64_t speaker_id = 0,
                 float speed = 1.0) const;

  /** Return an allocator for allocating memory
   */
  OrtAllocator *Allocator() const;

  const OfflineTtsPiperModelMetaData &GetMetaData() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_TTS_PIPER_MODEL_H_