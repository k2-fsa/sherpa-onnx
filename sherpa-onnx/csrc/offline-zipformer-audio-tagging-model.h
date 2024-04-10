// sherpa-onnx/csrc/offline-zipformer-audio-tagging-model.h
//
// Copyright (c)  2024  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_OFFLINE_ZIPFORMER_AUDIO_TAGGING_MODEL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_ZIPFORMER_AUDIO_TAGGING_MODEL_H_
#include <memory>
#include <utility>

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#include "onnxruntime_cxx_api.h"  // NOLINT
#include "sherpa-onnx/csrc/audio-tagging-model-config.h"

namespace sherpa_onnx {

/** This class implements the zipformer CTC model of the librispeech recipe
 * from icefall.
 *
 * See
 * https://github.com/k2-fsa/icefall/blob/master/egs/audioset/AT/zipformer/export-onnx.py
 */
class OfflineZipformerAudioTaggingModel {
 public:
  explicit OfflineZipformerAudioTaggingModel(
      const AudioTaggingModelConfig &config);

#if __ANDROID_API__ >= 9
  OfflineZipformerAudioTaggingModel(AAssetManager *mgr,
                                    const AudioTaggingModelConfig &config);
#endif

  ~OfflineZipformerAudioTaggingModel();

  /** Run the forward method of the model.
   *
   * @param features  A tensor of shape (N, T, C).
   * @param features_length  A 1-D tensor of shape (N,) containing number of
   *                         valid frames in `features` before padding.
   *                         Its dtype is int64_t.
   *
   * @return Return a tensor
   *  - probs: A 2-D tensor of shape (N, num_event_classes).
   */
  Ort::Value Forward(Ort::Value features, Ort::Value features_length) const;

  /** Return the number of event classes of the model
   */
  int32_t NumEventClasses() const;

  /** Return an allocator for allocating memory
   */
  OrtAllocator *Allocator() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_ZIPFORMER_AUDIO_TAGGING_MODEL_H_
