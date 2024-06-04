// sherpa-onnx/csrc/keyword-spotter.cc
//
// Copyright (c)  2023-2024  Xiaomi Corporation

#include "sherpa-onnx/csrc/keyword-spotter.h"

#include <assert.h>

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <memory>
#include <sstream>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/keyword-spotter-impl.h"

namespace sherpa_onnx {

std::string KeywordResult::AsJsonString() const {
  std::ostringstream os;
  os << "{";
  os << "\"start_time\":" << std::fixed << std::setprecision(2) << start_time
     << ", ";

  os << "\"keyword\""
     << ": ";
  os << "\"" << keyword << "\""
     << ", ";

  os << "\""
     << "timestamps"
     << "\""
     << ": ";
  os << "[";

  std::string sep = "";
  for (auto t : timestamps) {
    os << sep << std::fixed << std::setprecision(2) << t;
    sep = ", ";
  }
  os << "], ";

  os << "\""
     << "tokens"
     << "\""
     << ":";
  os << "[";

  sep = "";
  auto oldFlags = os.flags();
  for (const auto &t : tokens) {
    if (t.size() == 1 && static_cast<uint8_t>(t[0]) > 0x7f) {
      const uint8_t *p = reinterpret_cast<const uint8_t *>(t.c_str());
      os << sep << "\""
         << "<0x" << std::hex << std::uppercase << static_cast<uint32_t>(p[0])
         << ">"
         << "\"";
      os.flags(oldFlags);
    } else {
      os << sep << "\"" << t << "\"";
    }
    sep = ", ";
  }
  os << "]";
  os << "}";

  return os.str();
}

void KeywordSpotterConfig::Register(ParseOptions *po) {
  feat_config.Register(po);
  model_config.Register(po);

  po->Register("max-active-paths", &max_active_paths,
               "beam size used in modified beam search.");
  po->Register("num-trailing-blanks", &num_trailing_blanks,
               "The number of trailing blanks should have after the keyword.");
  po->Register("keywords-score", &keywords_score,
               "The bonus score for each token in context word/phrase.");
  po->Register("keywords-threshold", &keywords_threshold,
               "The acoustic threshold (probability) to trigger the keywords.");
  po->Register(
      "keywords-file", &keywords_file,
      "The file containing keywords, one word/phrase per line. For example: "
      "HELLO WORLD"
      "你好世界");
}

bool KeywordSpotterConfig::Validate() const {
  if (keywords_file.empty()) {
    SHERPA_ONNX_LOGE("Please provide --keywords-file.");
    return false;
  }

#ifndef SHERPA_ONNX_ENABLE_WASM_KWS
  // due to the limitations of the wasm file system,
  // keywords file will be packaged into the sherpa-onnx-wasm-kws-main.data file
  // Solution: take keyword_file variable is directly
  // parsed as a string of keywords
  if (!std::ifstream(keywords_file.c_str()).good()) {
    SHERPA_ONNX_LOGE("Keywords file '%s' does not exist.",
                     keywords_file.c_str());
    return false;
  }
#endif

  return model_config.Validate();
}

std::string KeywordSpotterConfig::ToString() const {
  std::ostringstream os;

  os << "KeywordSpotterConfig(";
  os << "feat_config=" << feat_config.ToString() << ", ";
  os << "model_config=" << model_config.ToString() << ", ";
  os << "max_active_paths=" << max_active_paths << ", ";
  os << "num_trailing_blanks=" << num_trailing_blanks << ", ";
  os << "keywords_score=" << keywords_score << ", ";
  os << "keywords_threshold=" << keywords_threshold << ", ";
  os << "keywords_file=\"" << keywords_file << "\")";

  return os.str();
}

KeywordSpotter::KeywordSpotter(const KeywordSpotterConfig &config)
    : impl_(KeywordSpotterImpl::Create(config)) {}

#if __ANDROID_API__ >= 9
KeywordSpotter::KeywordSpotter(AAssetManager *mgr,
                               const KeywordSpotterConfig &config)
    : impl_(KeywordSpotterImpl::Create(mgr, config)) {}
#endif

KeywordSpotter::~KeywordSpotter() = default;

std::unique_ptr<OnlineStream> KeywordSpotter::CreateStream() const {
  return impl_->CreateStream();
}

std::unique_ptr<OnlineStream> KeywordSpotter::CreateStream(
    const std::string &keywords) const {
  return impl_->CreateStream(keywords);
}

bool KeywordSpotter::IsReady(OnlineStream *s) const {
  return impl_->IsReady(s);
}

void KeywordSpotter::DecodeStreams(OnlineStream **ss, int32_t n) const {
  impl_->DecodeStreams(ss, n);
}

KeywordResult KeywordSpotter::GetResult(OnlineStream *s) const {
  return impl_->GetResult(s);
}

}  // namespace sherpa_onnx
