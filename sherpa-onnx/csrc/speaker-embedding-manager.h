// sherpa-onnx/csrc/speaker-embedding-manager.h
//
// Copyright (c)  2024  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_SPEAKER_EMBEDDING_MANAGER_H_
#define SHERPA_ONNX_CSRC_SPEAKER_EMBEDDING_MANAGER_H_

#include <memory>
#include <string>

namespace sherpa_onnx {

class SpeakerEmbeddingManager {
 public:
  // @param dim Embedding dimension.
  explicit SpeakerEmbeddingManager(int32_t dim);
  ~SpeakerEmbeddingManager();

  /* Add the embedding and name of a speaker to the manager.
   *
   * @param name Name of the speaker
   * @param p Pointer to the embedding. Its length is `dim`.
   * @return Return true if added successfully. Return false if it failed.
   *         At present, the only reason for a failure is that there is already
   *         a speaker with the same `name`.
   */
  bool Add(const std::string &name, const float *p) const;

  /* Remove a speaker by its name.
   *
   * @param name Name of the speaker to remove.
   * @return Return true if it is removed successfully. Return false
   *         if there is no such a speaker.
   */
  bool Remove(const std::string &name) const;

  /** It is for speaker identification.
   *
   * It computes the cosine similarity between and given embedding and all
   * other embeddings and find the embedding that has the largest score
   * and the score is above or equal to threshold. Return the speaker
   * name for the embedding if found; otherwise, it returns an empty string.
   *
   * @param p The input embedding.
   * @param threshold A value between 0 and 1.
   * @param If found, return the name of the speaker. Otherwise, return an
   *        empty string.
   */
  std::string Search(const float *p, float threshold) const;

  /* Check whether the input embedding matches the embedding of the input
   * speaker.
   *
   * It is for speaker verification.
   *
   * @param name The target speaker name.
   * @param p The input embedding to check.
   * @param threshold A value between 0 and 1.
   * @return Return true if it matches. Otherwise, it returns false.
   */
  bool Verify(const std::string &name, const float *p, float threshold) const;

  int32_t NumSpeakers() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_SPEAKER_EMBEDDING_MANAGER_H_
