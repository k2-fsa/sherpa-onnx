// sherpa-onnx/csrc/homophone-replacer.h
//
// Copyright (c)  2025  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_HOMOPHONE_REPLACER_H_
#define SHERPA_ONNX_CSRC_HOMOPHONE_REPLACER_H_

#include <memory>
#include <string>

#include "sherpa-onnx/csrc/parse-options.h"

namespace sherpa_onnx {

struct HomophoneReplacerConfig {
  std::string dict_dir;
  std::string lexicon;

  // comma separated fst files, e.g. a.fst,b.fst,c.fst
  std::string rule_fsts;

  bool debug;

  HomophoneReplacerConfig() = default;

  HomophoneReplacerConfig(const std::string &dict_dir,
                          const std::string &lexicon,
                          const std::string &rule_fsts, bool debug)
      : dict_dir(dict_dir),
        lexicon(lexicon),
        rule_fsts(rule_fsts),
        debug(debug) {}

  void Register(ParseOptions *po);
  bool Validate() const;

  std::string ToString() const;
};

class HomophoneReplacer {
 public:
  explicit HomophoneReplacer(const HomophoneReplacerConfig &config);

  template <typename Manager>
  HomophoneReplacer(Manager *mgr, const HomophoneReplacerConfig &config);

  ~HomophoneReplacer();

  std::string Apply(const std::string &text) const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_HOMOPHONE_REPLACER_H_
