// sherpa-onnx/csrc/online-recognizer.cc
//
// Copyright (c)  2023  Xiaomi Corporation
// Copyright (c)  2023  Pingfeng Luo

#include "sherpa-onnx/csrc/online-recognizer.h"

#include <assert.h>

#include <algorithm>
#include <iomanip>
#include <memory>
#include <sstream>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/online-recognizer-impl.h"

namespace sherpa_onnx {

std::string OnlineRecognizerResult::AsJsonString() const {
  std::ostringstream os;
  os << "{";
  os << "\"is_final\":" << (is_final ? "true" : "false") << ", ";
  os << "\"segment\":" << segment << ", ";
  os << "\"start_time\":" << std::fixed << std::setprecision(2) << start_time
     << ", ";

  os << "\"text\""
     << ": ";
  os << "\"" << text << "\""
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

void OnlineRecognizerConfig::Register(ParseOptions *po) {
  feat_config.Register(po);
  model_config.Register(po);
  endpoint_config.Register(po);
  lm_config.Register(po);

  po->Register("enable-endpoint", &enable_endpoint,
               "True to enable endpoint detection. False to disable it.");
  po->Register("max-active-paths", &max_active_paths,
               "beam size used in modified beam search.");
  po->Register("hotwords-score", &hotwords_score,
               "The bonus score for each token in context word/phrase. "
               "Used only when decoding_method is modified_beam_search");
  po->Register(
      "hotwords-file", &hotwords_file,
      "The file containing hotwords, one words/phrases per line, and for each"
      "phrase the bpe/cjkchar are separated by a space. For example: "
      "▁HE LL O ▁WORLD"
      "你 好 世 界");
  po->Register("decoding-method", &decoding_method,
               "decoding method,"
               "now support greedy_search and modified_beam_search.");
}

bool OnlineRecognizerConfig::Validate() const {
  if (decoding_method == "modified_beam_search" && !lm_config.model.empty()) {
    if (max_active_paths <= 0) {
      SHERPA_ONNX_LOGE("max_active_paths is less than 0! Given: %d",
                       max_active_paths);
      return false;
    }

    if (!lm_config.Validate()) {
      return false;
    }
  }

  if (!hotwords_file.empty() && decoding_method != "modified_beam_search") {
    SHERPA_ONNX_LOGE(
        "Please use --decoding-method=modified_beam_search if you"
        " provide --hotwords-file. Given --decoding-method=%s",
        decoding_method.c_str());
    return false;
  }

  return model_config.Validate();
}

std::string OnlineRecognizerConfig::ToString() const {
  std::ostringstream os;

  os << "OnlineRecognizerConfig(";
  os << "feat_config=" << feat_config.ToString() << ", ";
  os << "model_config=" << model_config.ToString() << ", ";
  os << "lm_config=" << lm_config.ToString() << ", ";
  os << "endpoint_config=" << endpoint_config.ToString() << ", ";
  os << "enable_endpoint=" << (enable_endpoint ? "True" : "False") << ", ";
  os << "max_active_paths=" << max_active_paths << ", ";
  os << "hotwords_score=" << hotwords_score << ", ";
  os << "hotwords_file=\"" << hotwords_file << "\", ";
  os << "decoding_method=\"" << decoding_method << "\")";

  return os.str();
}

OnlineRecognizer::OnlineRecognizer(const OnlineRecognizerConfig &config)
    : impl_(OnlineRecognizerImpl::Create(config)) {}

#if __ANDROID_API__ >= 9
OnlineRecognizer::OnlineRecognizer(AAssetManager *mgr,
                                   const OnlineRecognizerConfig &config)
    : impl_(OnlineRecognizerImpl::Create(mgr, config)) {}
#endif

OnlineRecognizer::~OnlineRecognizer() = default;

std::unique_ptr<OnlineStream> OnlineRecognizer::CreateStream() const {
  return impl_->CreateStream();
}

std::unique_ptr<OnlineStream> OnlineRecognizer::CreateStream(
    const std::string &hotwords) const {
  return impl_->CreateStream(hotwords);
}

bool OnlineRecognizer::IsReady(OnlineStream *s) const {
  return impl_->IsReady(s);
}

void OnlineRecognizer::DecodeStreams(OnlineStream **ss, int32_t n) const {
  impl_->DecodeStreams(ss, n);
}

OnlineRecognizerResult OnlineRecognizer::GetResult(OnlineStream *s) const {
  return impl_->GetResult(s);
}

bool OnlineRecognizer::IsEndpoint(OnlineStream *s) const {
  return impl_->IsEndpoint(s);
}

void OnlineRecognizer::Reset(OnlineStream *s) const { impl_->Reset(s); }

}  // namespace sherpa_onnx
