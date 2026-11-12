// sherpa-onnx/csrc/forced-aligner.cc
//
// Copyright (c)  2026  Xiaomi Corporation

#include "sherpa-onnx/csrc/forced-aligner.h"

#include <iomanip>
#include <memory>
#include <sstream>
#include <string>

#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/forced-aligner-impl.h"
#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {

void ForcedAlignmentOptions::Register(ParseOptions *po) {
  po->Register("beam", &beam,
               "Decoding beam used by the alignment search. Larger is more "
               "accurate and slower.");
  po->Register("retry-beam", &retry_beam,
               "Beam used to retry failed alignments. 0 disables retry.");
  po->Register("transition-scale", &transition_scale,
               "Transition-probability scale [relative to acoustics]");
  po->Register("self-loop-scale", &self_loop_scale,
               "Scale of self-loop versus non-self-loop log probs [relative "
               "to acoustics]");
  po->Register("acoustic-scale", &acoustic_scale,
               "Scaling factor for acoustic likelihoods");
}

std::string ForcedAlignmentOptions::ToString() const {
  std::ostringstream os;
  os << "ForcedAlignmentOptions(";
  os << "beam=" << beam << ", ";
  os << "retry_beam=" << retry_beam << ", ";
  os << "transition_scale=" << transition_scale << ", ";
  os << "self_loop_scale=" << self_loop_scale << ", ";
  os << "acoustic_scale=" << acoustic_scale << ")";
  return os.str();
}

void ForcedAlignerConfig::Register(ParseOptions *po) {
  po->Register(
      "model-dir", &model_dir,
      "Directory of the acoustic model. It must contain:\n"
      "  meta.json     feature chain parameters (mfcc/pitch/cmvn, silence\n"
      "                probability, OOV phone)\n"
      "  final.alimdl  the alignment model, kaldi format: TransitionModel +\n"
      "                AmDiagGmm. NOTE: this is final.alimdl, not final.mdl\n"
      "  tree          the context dependency tree (kaldi ContextDependency)\n"
      "  lda.mat       the LDA transform of the feature chain; required for\n"
      "                models whose chain ends in splice+LDA (most MFA GMM\n"
      "                models), optional otherwise\n"
      "  phones.txt    phone symbol table ('symbol id' per line); when it is\n"
      "                missing the phones list of meta.json is used instead\n"
      "Files not used by the aligner (final.mdl, phone_lm.fst,\n"
      "graphemes.txt, phone_pdf.counts, ...) may be present and are ignored.");
  po->Register("lexicon", &lexicon,
               "Pronunciation dictionary used to build the lexicon FST. "
               "Please see forced-aligner.h for the format.");
  po->Register("debug", &debug,
               "true to print debug information (model info while loading, "
               "per-utterance alignment info in Compute).");
  options.Register(po);
}

bool ForcedAlignerConfig::Validate() const {
  if (model_dir.empty()) {
    SHERPA_ONNX_LOGE("model-dir is empty");
    return false;
  }
  if (lexicon.empty()) {
    SHERPA_ONNX_LOGE("lexicon is empty");
    return false;
  }
  if (!FileExists(model_dir + "/meta.json")) {
    SHERPA_ONNX_LOGE("no meta.json in model-dir: '%s'", model_dir.c_str());
    return false;
  }
  if (!FileExists(lexicon)) {
    SHERPA_ONNX_LOGE("lexicon does not exist: '%s'", lexicon.c_str());
    return false;
  }
  return true;
}

std::string ForcedAlignerConfig::ToString() const {
  std::ostringstream os;
  os << "ForcedAlignerConfig(";
  os << "model_dir=\"" << model_dir << "\", ";
  os << "lexicon=\"" << lexicon << "\", ";
  os << "debug=" << (debug ? "True" : "False") << ", ";
  os << options.ToString() << ")";
  return os.str();
}

// Inside a Praat text tier, a double quote is written as two double quotes.
static std::string EscapeTextGridText(const std::string &s) {
  std::string out;
  out.reserve(s.size());
  for (char c : s) {
    if (c == '"') {
      out += "\"\"";
    } else {
      out += c;
    }
  }
  return out;
}

std::string ForcedAlignmentResult::ToTextGrid() const {
  std::ostringstream os;
  os << std::fixed << std::setprecision(4);

  os << "File type = \"ooTextFile\"\n";
  os << "Object class = \"TextGrid\"\n";
  os << "\n";
  os << "xmin = 0\n";
  os << "xmax = " << duration << "\n";
  os << "tiers? <exists>\n";
  os << "size = " << tiers.size() << "\n";
  os << "item []:\n";

  for (size_t i = 0; i != tiers.size(); ++i) {
    const auto &tier = tiers[i];
    os << "    item [" << i + 1 << "]:\n";
    os << "        class = \"IntervalTier\"\n";
    os << "        name = \"" << EscapeTextGridText(tier.name) << "\"\n";
    os << "        xmin = 0\n";
    os << "        xmax = " << duration << "\n";
    os << "        intervals: size = " << tier.intervals.size() << "\n";
    for (size_t j = 0; j != tier.intervals.size(); ++j) {
      const auto &iv = tier.intervals[j];
      os << "        intervals [" << j + 1 << "]:\n";
      os << "            xmin = " << iv.start << "\n";
      os << "            xmax = " << iv.end << "\n";
      os << "            text = \"" << EscapeTextGridText(iv.text) << "\"\n";
    }
  }
  return os.str();
}

ForcedAligner::ForcedAligner(const ForcedAlignerConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

ForcedAligner::~ForcedAligner() = default;

std::unique_ptr<OfflineStream> ForcedAligner::CreateStream() const {
  return impl_->CreateStream();
}

ForcedAlignmentResult ForcedAligner::Compute(const OfflineStream *stream,
                                             const std::string &text) const {
  return impl_->Compute(stream, text);
}

}  // namespace sherpa_onnx
