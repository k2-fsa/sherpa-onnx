// sherpa-onnx/csrc/audio-tagging-label-file.h
//
// Copyright (c)  2024  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_AUDIO_TAGGING_LABEL_FILE_H_
#define SHERPA_ONNX_CSRC_AUDIO_TAGGING_LABEL_FILE_H_

#include <istream>
#include <string>
#include <vector>

namespace sherpa_onnx {

class AudioTaggingLabels {
 public:
  explicit AudioTaggingLabels(const std::string &filename);

  // Return the event name for the given index.
  // The returned reference is valid as long as this object is alive
  const std::string &GetEventName(int32_t index) const;
  int32_t NumEventClasses() const { return names_.size(); }

 private:
  void Init(std::istream &is);

 private:
  std::vector<std::string> names_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_AUDIO_TAGGING_LABEL_FILE_H_
