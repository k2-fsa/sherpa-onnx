// sherpa-onnx/csrc/nomo-pvad-model-config.h
//
// Copyright (c)  2026  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_NOMO_PVAD_MODEL_CONFIG_H_
#define SHERPA_ONNX_CSRC_NOMO_PVAD_MODEL_CONFIG_H_

#include <string>
#include <vector>

#include "sherpa-onnx/csrc/parse-options.h"

namespace sherpa_onnx {

/** Config for the NOMO-PVAD target-speaker VAD model.
 *
 * See https://huggingface.co/TuyaNomo/nomo-pvad
 *
 * Unlike Silero VAD and TEN VAD, which answer "is anybody speaking?", this
 * model answers "is the *enrolled* speaker speaking?". You therefore have to
 * provide the 192-d speaker embedding of the target speaker, either
 * programmatically via `speaker_embedding` or through a text file via
 * `speaker_embedding_file`.
 *
 * The embedding is usually computed once from an enrollment clip with
 * a speaker embedding extractor, e.g., the 3d-speaker ERes2NetV2 model that
 * `sherpa-onnx` already supports.
 */
struct NomoPvadModelConfig {
  std::string model;

  /** 192-d enrollment embedding of the target speaker.
   *
   * It is L2-normalized inside the model, so normalization is not required
   * from the caller.
   */
  std::vector<float> speaker_embedding;

  /** Path to a text file containing the enrollment embedding as whitespace-
   * or comma-separated floats. It is only used when `speaker_embedding` is
   * empty, so that the model can also be driven from the command line.
   */
  std::string speaker_embedding_file;

  /** Probability above which a chunk counts as the target speaker talking.
   *
   * The model card suggests 0.7 - 0.85 if a false acceptance is expensive
   * and 0.5 - 0.6 if recall matters more.
   */
  float threshold = 0.7;

  float min_silence_duration = 0.5;  // in seconds
  float min_speech_duration = 0.25;  // in seconds

  /** If a speech segment is longer than this value, then we increase
   * the threshold to 0.9. After finishing detecting the segment,
   * the threshold value is reset to its original value.
   */
  float max_speech_duration = 20;  // in seconds

  /** Threshold for transitioning from speech to silence.
   *
   * A value smaller than `threshold` creates a hysteresis band so that the
   * detector does not flap around the decision boundary. If left negative,
   * we use max(threshold - 0.15, 0.01).
   */
  float neg_threshold = -1;

  NomoPvadModelConfig() = default;

  void Register(ParseOptions *po);

  bool Validate() const;

  // Like Validate(), for a model file given by a resource manager (Android
  // asset manager / OHOS native resource manager): `model` and
  // `speaker_embedding_file` are paths relative to the resource root.
  template <typename Manager>
  bool Validate(Manager *mgr) const;

  std::string ToString() const;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_NOMO_PVAD_MODEL_CONFIG_H_
