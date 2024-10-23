// sherpa-onnx/csrc/online-recognizer-impl.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/online-recognizer-impl.h"

#include <utility>

#if __ANDROID_API__ >= 9
#include <strstream>

#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#include "fst/extensions/far/far.h"
#include "kaldifst/csrc/kaldi-fst-io.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/online-recognizer-ctc-impl.h"
#include "sherpa-onnx/csrc/online-recognizer-paraformer-impl.h"
#include "sherpa-onnx/csrc/online-recognizer-transducer-impl.h"
#include "sherpa-onnx/csrc/online-recognizer-transducer-nemo-impl.h"
#include "sherpa-onnx/csrc/onnx-utils.h"
#include "sherpa-onnx/csrc/text-utils.h"

namespace sherpa_onnx {

std::unique_ptr<OnlineRecognizerImpl> OnlineRecognizerImpl::Create(
    const OnlineRecognizerConfig &config) {
  if (!config.model_config.transducer.encoder.empty()) {
    Ort::Env env(ORT_LOGGING_LEVEL_ERROR);

    Ort::SessionOptions sess_opts;
    sess_opts.SetIntraOpNumThreads(1);
    sess_opts.SetInterOpNumThreads(1);

    auto decoder_model = ReadFile(config.model_config.transducer.decoder);
    auto sess = std::make_unique<Ort::Session>(env, decoder_model.data(),
                                               decoder_model.size(), sess_opts);

    size_t node_count = sess->GetOutputCount();

    if (node_count == 1) {
      return std::make_unique<OnlineRecognizerTransducerImpl>(config);
    } else {
      return std::make_unique<OnlineRecognizerTransducerNeMoImpl>(config);
    }
  }

  if (!config.model_config.paraformer.encoder.empty()) {
    return std::make_unique<OnlineRecognizerParaformerImpl>(config);
  }

  if (!config.model_config.wenet_ctc.model.empty() ||
      !config.model_config.zipformer2_ctc.model.empty() ||
      !config.model_config.nemo_ctc.model.empty()) {
    return std::make_unique<OnlineRecognizerCtcImpl>(config);
  }

  SHERPA_ONNX_LOGE("Please specify a model");
  exit(-1);
}

#if __ANDROID_API__ >= 9
std::unique_ptr<OnlineRecognizerImpl> OnlineRecognizerImpl::Create(
    AAssetManager *mgr, const OnlineRecognizerConfig &config) {
  if (!config.model_config.transducer.encoder.empty()) {
    Ort::Env env(ORT_LOGGING_LEVEL_ERROR);

    Ort::SessionOptions sess_opts;
    sess_opts.SetIntraOpNumThreads(1);
    sess_opts.SetInterOpNumThreads(1);

    auto decoder_model = ReadFile(mgr, config.model_config.transducer.decoder);
    auto sess = std::make_unique<Ort::Session>(env, decoder_model.data(),
                                               decoder_model.size(), sess_opts);

    size_t node_count = sess->GetOutputCount();

    if (node_count == 1) {
      return std::make_unique<OnlineRecognizerTransducerImpl>(mgr, config);
    } else {
      return std::make_unique<OnlineRecognizerTransducerNeMoImpl>(mgr, config);
    }
  }

  if (!config.model_config.paraformer.encoder.empty()) {
    return std::make_unique<OnlineRecognizerParaformerImpl>(mgr, config);
  }

  if (!config.model_config.wenet_ctc.model.empty() ||
      !config.model_config.zipformer2_ctc.model.empty() ||
      !config.model_config.nemo_ctc.model.empty()) {
    return std::make_unique<OnlineRecognizerCtcImpl>(mgr, config);
  }

  SHERPA_ONNX_LOGE("Please specify a model");
  exit(-1);
}
#endif

OnlineRecognizerImpl::OnlineRecognizerImpl(const OnlineRecognizerConfig &config)
    : config_(config) {
  if (!config.rule_fsts.empty()) {
    std::vector<std::string> files;
    SplitStringToVector(config.rule_fsts, ",", false, &files);
    itn_list_.reserve(files.size());
    for (const auto &f : files) {
      if (config.model_config.debug) {
        SHERPA_ONNX_LOGE("rule fst: %s", f.c_str());
      }
      itn_list_.push_back(std::make_unique<kaldifst::TextNormalizer>(f));
    }
  }

  if (!config.rule_fars.empty()) {
    if (config.model_config.debug) {
      SHERPA_ONNX_LOGE("Loading FST archives");
    }
    std::vector<std::string> files;
    SplitStringToVector(config.rule_fars, ",", false, &files);

    itn_list_.reserve(files.size() + itn_list_.size());

    for (const auto &f : files) {
      if (config.model_config.debug) {
        SHERPA_ONNX_LOGE("rule far: %s", f.c_str());
      }
      std::unique_ptr<fst::FarReader<fst::StdArc>> reader(
          fst::FarReader<fst::StdArc>::Open(f));
      for (; !reader->Done(); reader->Next()) {
        std::unique_ptr<fst::StdConstFst> r(
            fst::CastOrConvertToConstFst(reader->GetFst()->Copy()));

        itn_list_.push_back(
            std::make_unique<kaldifst::TextNormalizer>(std::move(r)));
      }
    }

    if (config.model_config.debug) {
      SHERPA_ONNX_LOGE("FST archives loaded!");
    }
  }
}

#if __ANDROID_API__ >= 9
OnlineRecognizerImpl::OnlineRecognizerImpl(AAssetManager *mgr,
                                           const OnlineRecognizerConfig &config)
    : config_(config) {
  if (!config.rule_fsts.empty()) {
    std::vector<std::string> files;
    SplitStringToVector(config.rule_fsts, ",", false, &files);
    itn_list_.reserve(files.size());
    for (const auto &f : files) {
      if (config.model_config.debug) {
        SHERPA_ONNX_LOGE("rule fst: %s", f.c_str());
      }
      auto buf = ReadFile(mgr, f);
      std::istrstream is(buf.data(), buf.size());
      itn_list_.push_back(std::make_unique<kaldifst::TextNormalizer>(is));
    }
  }

  if (!config.rule_fars.empty()) {
    std::vector<std::string> files;
    SplitStringToVector(config.rule_fars, ",", false, &files);
    itn_list_.reserve(files.size() + itn_list_.size());

    for (const auto &f : files) {
      if (config.model_config.debug) {
        SHERPA_ONNX_LOGE("rule far: %s", f.c_str());
      }

      auto buf = ReadFile(mgr, f);

      std::unique_ptr<std::istream> s(
          new std::istrstream(buf.data(), buf.size()));

      std::unique_ptr<fst::FarReader<fst::StdArc>> reader(
          fst::FarReader<fst::StdArc>::Open(std::move(s)));

      for (; !reader->Done(); reader->Next()) {
        std::unique_ptr<fst::StdConstFst> r(
            fst::CastOrConvertToConstFst(reader->GetFst()->Copy()));

        itn_list_.push_back(
            std::make_unique<kaldifst::TextNormalizer>(std::move(r)));
      }  // for (; !reader->Done(); reader->Next())
    }    // for (const auto &f : files)
  }      // if (!config.rule_fars.empty())
}
#endif

std::string OnlineRecognizerImpl::ApplyInverseTextNormalization(
    std::string text) const {
  if (!itn_list_.empty()) {
    for (const auto &tn : itn_list_) {
      text = tn->Normalize(text);
    }
  }

  return text;
}

}  // namespace sherpa_onnx
