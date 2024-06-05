// sherpa-onnx/csrc/audio-tagging-label-file.cc
//
// Copyright (c)  2024  Xiaomi Corporation

#include "sherpa-onnx/csrc/audio-tagging-label-file.h"

#include <fstream>
#include <sstream>
#include <string>

#if __ANDROID_API__ >= 9
#include <strstream>

#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/onnx-utils.h"
#include "sherpa-onnx/csrc/text-utils.h"

namespace sherpa_onnx {

AudioTaggingLabels::AudioTaggingLabels(const std::string &filename) {
  std::ifstream is(filename);
  Init(is);
}

#if __ANDROID_API__ >= 9
AudioTaggingLabels::AudioTaggingLabels(AAssetManager *mgr,
                                       const std::string &filename) {
  auto buf = ReadFile(mgr, filename);
  std::istrstream is(buf.data(), buf.size());
  Init(is);
}
#endif

// Format of a label file
/*
index,mid,display_name
0,/m/09x0r,"Speech"
1,/m/05zppz,"Male speech, man speaking"
*/
void AudioTaggingLabels::Init(std::istream &is) {
  std::string line;
  std::getline(is, line);  // skip the header

  std::string index;
  std::string tmp;
  std::string name;

  while (std::getline(is, line)) {
    index.clear();
    name.clear();
    std::istringstream input2(line);

    std::getline(input2, index, ',');
    std::getline(input2, tmp, ',');
    std::getline(input2, name);

    std::size_t pos{};
    int32_t i = std::stoi(index, &pos);
    if (index.size() == 0 || pos != index.size()) {
      SHERPA_ONNX_LOGE("Invalid line: %s", line.c_str());
      exit(-1);
    }

    if (i != static_cast<int32_t>(names_.size())) {
      SHERPA_ONNX_LOGE(
          "Index should be sorted and contiguous. Expected index: %d, given: "
          "%d.",
          static_cast<int32_t>(names_.size()), i);
    }
    if (name.empty() || name.front() != '"' || name.back() != '"') {
      SHERPA_ONNX_LOGE("Invalid line: %s", line.c_str());
      exit(-1);
    }

    names_.emplace_back(name.begin() + 1, name.end() - 1);
  }
}

const std::string &AudioTaggingLabels::GetEventName(int32_t index) const {
  return names_.at(index);
}

}  // namespace sherpa_onnx
