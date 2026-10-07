// sherpa-onnx/csrc/offline-speaker-segmentation-sortformer-model.h
//
// Copyright (c)  2026  Silvio Tomatis
#ifndef SHERPA_ONNX_CSRC_OFFLINE_SPEAKER_SEGMENTATION_SORTFORMER_MODEL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_SPEAKER_SEGMENTATION_SORTFORMER_MODEL_H_

#include <memory>
#include <utility>

#include "onnxruntime_cxx_api.h"  // NOLINT
#include "sherpa-onnx/csrc/offline-speaker-segmentation-model-config.h"
#include "sherpa-onnx/csrc/offline-speaker-segmentation-sortformer-model-meta-data.h"

namespace sherpa_onnx {

class OfflineSpeakerSegmentationSortformerModel {
 public:
  explicit OfflineSpeakerSegmentationSortformerModel(
      const OfflineSpeakerSegmentationModelConfig &config);

  template <typename Manager>
  OfflineSpeakerSegmentationSortformerModel(
      Manager *mgr, const OfflineSpeakerSegmentationModelConfig &config);

  ~OfflineSpeakerSegmentationSortformerModel();

  const OfflineSpeakerSegmentationSortformerModelMetaData &GetModelMetaData()
      const;

  OrtAllocator *Allocator() const;

  /**
   * @param features A 3-D float tensor of shape (1, T, num_mel_bins).
   *                 T must be a multiple of subsampling_factor.
   * @param cached_embeds A 3-D float tensor of shape (1, C, hidden_size),
   *                      containing the speaker cache and FIFO queue. C can
   *                      be 0.
   * @return Return a pair:
   *   - probs: (1, (C + T / subsampling_factor) * subsampling_factor,
   *             num_speakers), speaker activity probabilities
   *   - chunk_embeds: (1, T / subsampling_factor, hidden_size)
   */
  std::pair<Ort::Value, Ort::Value> Forward(Ort::Value features,
                                            Ort::Value cached_embeds) const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_SPEAKER_SEGMENTATION_SORTFORMER_MODEL_H_
