// sherpa-onnx/csrc/offline-recognizer-impl.h
//
// Copyright (c)  2023  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_OFFLINE_RECOGNIZER_IMPL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_RECOGNIZER_IMPL_H_

#include <memory>
#include <string>
#include <vector>

#include "kaldifst/csrc/text-normalizer.h"
#include "sherpa-onnx/csrc/homophone-replacer.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/offline-recognizer.h"
#include "sherpa-onnx/csrc/offline-stream.h"

namespace sherpa_onnx {

class OfflineRecognizerImpl {
 public:
  explicit OfflineRecognizerImpl(const OfflineRecognizerConfig &config);

  static std::unique_ptr<OfflineRecognizerImpl> Create(
      const OfflineRecognizerConfig &config);

  template <typename Manager>
  OfflineRecognizerImpl(Manager *mgr, const OfflineRecognizerConfig &config);

  template <typename Manager>
  static std::unique_ptr<OfflineRecognizerImpl> Create(
      Manager *mgr, const OfflineRecognizerConfig &config);

  virtual ~OfflineRecognizerImpl() = default;

  virtual std::unique_ptr<OfflineStream> CreateStream(
      const std::string &hotwords) const {
    SHERPA_ONNX_LOGE("Only transducer models support contextual biasing.");
    SHERPA_ONNX_EXIT(-1);
  }

  virtual std::unique_ptr<OfflineStream> CreateStream() const = 0;

  virtual void DecodeStreams(OfflineStream **ss, int32_t n) const = 0;

  virtual void SetConfig(const OfflineRecognizerConfig &config);

  virtual OfflineRecognizerConfig GetConfig() const = 0;

  // Inject an Ort::RunOptions config entry (e.g.,
  // "memory.enable_memory_arena_shrinkage" = "cpu:0") that will be applied
  // to all subsequent Session::Run calls of this recognizer.
  //
  // The default implementation logs a warning and ignores the setting.
  // Concrete recognizers that route their inference through a model class
  // owning the Ort::Session(s) should override this and forward the entry
  // to the underlying model.
  virtual void SetRunOptionsConfigEntry(const std::string &key,
                                        const std::string &value);

  std::string ApplyInverseTextNormalization(std::string text) const;

  std::string ApplyHomophoneReplacer(std::string text) const;

 protected:
  OfflineRecognizerConfig config_;
  // for inverse text normalization. Used only if
  // config.rule_fsts is not empty or
  // config.rule_fars is not empty
  std::vector<std::unique_ptr<kaldifst::TextNormalizer>> itn_list_;
  std::unique_ptr<HomophoneReplacer> hr_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_RECOGNIZER_IMPL_H_
