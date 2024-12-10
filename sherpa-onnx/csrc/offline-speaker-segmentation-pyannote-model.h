// sherpa-onnx/csrc/offline-speaker-segmentation-pyannote-model.h
//
// Copyright (c)  2024  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_OFFLINE_SPEAKER_SEGMENTATION_PYANNOTE_MODEL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_SPEAKER_SEGMENTATION_PYANNOTE_MODEL_H_

#include <memory>

#include "onnxruntime_cxx_api.h"  // NOLINT
#include "sherpa-onnx/csrc/offline-speaker-segmentation-model-config.h"
#include "sherpa-onnx/csrc/offline-speaker-segmentation-pyannote-model-meta-data.h"

namespace sherpa_onnx {

class OfflineSpeakerSegmentationPyannoteModel {
 public:
  explicit OfflineSpeakerSegmentationPyannoteModel(
      const OfflineSpeakerSegmentationModelConfig &config);

  template <typename Manager>
  OfflineSpeakerSegmentationPyannoteModel(
      Manager *mgr, const OfflineSpeakerSegmentationModelConfig &config);

  ~OfflineSpeakerSegmentationPyannoteModel();

  const OfflineSpeakerSegmentationPyannoteModelMetaData &GetModelMetaData()
      const;

  /**
   * @param x A 3-D float tensor of shape (batch_size, 1, num_samples)
   * @return Return a float tensor of
   *         shape (batch_size, num_frames, num_speakers). Note that
   *         num_speakers here uses powerset encoding.
   */
  Ort::Value Forward(Ort::Value x) const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_SPEAKER_SEGMENTATION_PYANNOTE_MODEL_H_
