// sherpa-onnx/csrc/keyword-spotter.cc
//
// Copyright (c)  2023-2024  Xiaomi Corporation

#include "sherpa-onnx/csrc/keyword-spotter.h"

#include <assert.h>

#include <algorithm>
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
  endpoint_config.Register(po);

  po->Register("enable-endpoint", &enable_endpoint,
               "True to enable endpoint detection. False to disable it.");
  po->Register("max-active-paths", &max_active_paths,
               "beam size used in modified beam search.");
  po->Register("num-tailing-blanks", &num_tailing_blanks,
               "The number of tailing blanks should have after the keyword.");
  po->Register("keywords-score", &keywords_score,
               "The bonus score for each token in context word/phrase.");
  po->Register("keywords-threshold", &keywords_threshold,
               "The acoustic threshold (probability) to trigger the keywords.");
  po->Register(
      "keywords-file", &hotwords_file,
      "The file containing keywords, one words/phrases per line, and for each"
      "phrase the bpe/cjkchar are separated by a space. For example: "
      "▁HE LL O ▁WORLD"
      "你 好 世 界");
}

bool KeywordSpotterConfig::Validate() const {
  if (keywords_file.empty()) {
    SHERPA_ONNX_LOGE("Please provide --keywords-file.");
    return false;
  }

  return model_config.Validate();
}

std::string KeywordSpotterConfig::ToString() const {
  std::ostringstream os;

  os << "KeywordSpotterConfig(";
  os << "feat_config=" << feat_config.ToString() << ", ";
  os << "model_config=" << model_config.ToString() << ", ";
  os << "endpoint_config=" << endpoint_config.ToString() << ", ";
  os << "enable_endpoint=" << (enable_endpoint ? "True" : "False") << ", ";
  os << "max_active_paths=" << max_active_paths << ", ";
  os << "num_tailing_blanks=" << num_tailing_blanks << ", ";
  os << "keywords_score=" << keywords_score << ", ";
  os << "keywords_threshold=" << keywords_shreshold << ", ";
  os << "keywords_file=\"" << keywords_file << "\", ";

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

bool KeywordSpotter::IsReady(OnlineStream *s) const {
  return impl_->IsReady(s);
}

void KeywordSpotter::DecodeStreams(OnlineStream **ss, int32_t n) const {
  impl_->DecodeStreams(ss, n);
}

KeywordResult KeywordSpotter::GetResult(OnlineStream *s) const {
  return impl_->GetResult(s);
}

bool KeywordSpotter::IsEndpoint(OnlineStream *s) const {
  return impl_->IsEndpoint(s);
}

void KeywordSpotter::Reset(OnlineStream *s) const { impl_->Reset(s); }

}  // namespace sherpa_onnx
