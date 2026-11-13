// sherpa-onnx/csrc/forced-aligner-impl.cc
//
// Copyright (c)  2026  Xiaomi Corporation

#include "sherpa-onnx/csrc/forced-aligner-impl.h"

#include <algorithm>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "kaldi-decoder/csrc/decoder-wrappers.h"
#include "kaldi-hmm-gmm/csrc/decodable-am-diag-gmm.h"
#include "kaldi-hmm-gmm/csrc/eigen.h"
#include "kaldi-hmm-gmm/csrc/hmm-utils.h"
#include "kaldi-hmm-gmm/csrc/training-graph-compiler.h"
#include "kaldi-native-fbank/csrc/feature-delta.h"
#include "kaldi-native-fbank/csrc/feature-matrix.h"
#include "kaldi-native-fbank/csrc/feature-splice.h"
#include "kaldi_native_io/csrc/kaldi-io.h"
#include "nlohmann/json.hpp"
#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {

// ---------------------------------------------------------------------------
// meta.json parsing
// ---------------------------------------------------------------------------

static float JsonFloat(const nlohmann::json &j, const char *key, float dflt) {
  return j.contains(key) && j.at(key).is_number() ? j.at(key).get<float>()
                                                  : dflt;
}

static int32_t JsonInt(const nlohmann::json &j, const char *key, int32_t dflt) {
  return j.contains(key) && j.at(key).is_number_integer()
             ? j.at(key).get<int32_t>()
             : dflt;
}

static bool JsonBool(const nlohmann::json &j, const char *key, bool dflt) {
  return j.contains(key) && j.at(key).is_boolean() ? j.at(key).get<bool>()
                                                   : dflt;
}

// Parses the model's meta.json. The relevant part of the format:
//
//   {
//     "features": {                     // feature extraction parameters
//       "type": "mfcc",
//       "sample_frequency": 16000,
//       "frame_shift": 10,             // ms
//       "frame_length": 25,            // ms
//       "snip_edges": false,
//       "dither": 0.0001,
//       "use_energy": true,
//       "energy_floor": 1.0,
//       "raw_energy": true,
//       "num_mel_bins": 23,
//       "num_coefficients": 13,
//       "cepstral_lifter": 22,
//       "low_frequency": 20,
//       "high_frequency": 7800,
//       "preemphasis_coefficient": 0.97,
//       "use_pitch": false,            // concat 3 processed-pitch dims
//       "min_f0": 50, "max_f0": 500,
//       "delta_pitch": 0.005,
//       "penalty_factor": 0.1,
//       "uses_cmvn": true,             // utterance-level, mean only
//       "uses_deltas": false,          // delta features when no lda.mat
//       "uses_splices": false,         // chain ends in splice + LDA
//       "splice_left_context": 3,
//       "splice_right_context": 3
//     },
//     "silence_probability": 0.17,     // optional silence in the lexicon FST
//     "oov_phone": "spn",              // pronunciation of unknown words
//     "optional_silence_phone": "sil",
//     "phones": ["sil", "spn", "..."]  // phone table fallback for phones.txt
//   }
//
// All fields are optional; missing fields get kaldi defaults. Unsupported
// model features (i-vectors, FMLLR, subsampling, ...) raise a
// std::runtime_error naming the offending field.
static FaModelMeta ParseFaModelMeta(const std::string &filename) {
  std::ifstream is = OpenInputFile(filename);
  if (!is) {
    throw std::runtime_error("Failed to open meta.json: '" + filename + "'");
  }
  nlohmann::json j;
  try {
    is >> j;
  } catch (const std::exception &e) {
    throw std::runtime_error("Failed to parse '" + filename + "': " + e.what());
  }

  FaModelMeta meta;
  if (!j.contains("features") || !j.at("features").is_object()) {
    throw std::runtime_error("meta.json has no \"features\" section: '" +
                             filename + "'");
  }
  const auto &f = j.at("features");

  if (f.contains("type") && f.at("type").is_string() &&
      f.at("type").get<std::string>() != "mfcc") {
    throw std::runtime_error(
        "ForcedAligner: meta.json uses unsupported feature type '" +
        f.at("type").get<std::string>() +
        "' (only MFCC models are supported yet). "
        "Please file an issue at https://github.com/k2-fsa/sherpa-onnx");
  }

  // Features this aligner cannot model. uses_speaker_adaptation is NOT
  // listed: it describes training-time adaptation and every GMM
  // alignment model carries it; the semantic baseline
  // (kaldi-hmm-gmm/tools/align_mfa.py) ignores it as well.
  for (const char *key : {"uses_i_vector", "subsample_feats"}) {
    if (JsonBool(f, key, false)) {
      throw std::runtime_error(
          std::string("ForcedAligner: meta.json uses unsupported feature '") +
          key +
          "' (i-vector/FMLLR models are not supported yet). "
          "Please file an issue at https://github.com/k2-fsa/sherpa-onnx");
    }
  }
  for (auto it = f.begin(); it != f.end(); ++it) {
    const std::string &key = it.key();
    if ((key.rfind("ivector_", 0) == 0 || key.rfind("fmllr_", 0) == 0 ||
         key.rfind("vad_", 0) == 0) &&
        !it.value().is_null() &&
        !(it.value().is_boolean() && !it.value().get<bool>())) {
      throw std::runtime_error(
          "ForcedAligner: meta.json uses unsupported feature '" + key +
          "' (i-vector/FMLLR models are not supported yet). "
          "Please file an issue at https://github.com/k2-fsa/sherpa-onnx");
    }
  }
  for (const char *key : {"sliding_cmvn", "use_voicing", "use_delta_pitch",
                          "normalize_pitch", "adaptive_pitch_range"}) {
    if (JsonBool(f, key, false)) {
      SHERPA_ONNX_LOGE(
          "meta.json sets %s; it is only partially supported and may be "
          "ignored (utterance CMVN / standard 3-dim pitch are used)",
          key);
    }
  }

  FaFeatureConfig &feat = meta.features;
  feat.sample_frequency = JsonFloat(f, "sample_frequency", 16000);
  feat.frame_shift = JsonFloat(f, "frame_shift", 10);
  feat.frame_length = JsonFloat(f, "frame_length", 25);
  feat.snip_edges = JsonBool(f, "snip_edges", true);
  feat.dither = JsonFloat(f, "dither", 0.0001f);
  feat.use_energy = JsonBool(f, "use_energy", false);
  feat.energy_floor = JsonFloat(f, "energy_floor", 1.0f);
  feat.raw_energy = JsonBool(f, "raw_energy", true);
  feat.num_mel_bins = JsonInt(f, "num_mel_bins", 23);
  feat.num_coefficients = JsonInt(f, "num_coefficients", 13);
  feat.cepstral_lifter = JsonFloat(f, "cepstral_lifter", 22);
  feat.low_frequency = JsonFloat(f, "low_frequency", 20);
  feat.high_frequency = JsonFloat(f, "high_frequency", 7800);
  feat.preemphasis_coefficient = JsonFloat(f, "preemphasis_coefficient", 0.97f);
  feat.use_pitch = JsonBool(f, "use_pitch", false);
  feat.min_f0 = JsonFloat(f, "min_f0", 50);
  feat.max_f0 = JsonFloat(f, "max_f0", 500);
  feat.delta_pitch = JsonFloat(f, "delta_pitch", 0.005f);
  feat.penalty_factor = JsonFloat(f, "penalty_factor", 0.1f);
  feat.uses_cmvn = JsonBool(f, "uses_cmvn", true);

  meta.uses_deltas = JsonBool(f, "uses_deltas", false);
  meta.uses_splices = JsonBool(f, "uses_splices", false);
  meta.splice_left_context = JsonInt(f, "splice_left_context", 3);
  meta.splice_right_context = JsonInt(f, "splice_right_context", 3);

  meta.silence_probability = JsonFloat(j, "silence_probability", 0.5);
  if (j.contains("oov_phone") && j.at("oov_phone").is_string()) {
    meta.oov_phone = j.at("oov_phone").get<std::string>();
  }
  if (j.contains("optional_silence_phone") &&
      j.at("optional_silence_phone").is_string()) {
    meta.optional_silence_phone =
        j.at("optional_silence_phone").get<std::string>();
  }
  if (j.contains("phones") && j.at("phones").is_array()) {
    for (const auto &p : j.at("phones")) {
      if (p.is_string()) {
        meta.phones.push_back(p.get<std::string>());
      }
    }
  }
  return meta;
}

// ---------------------------------------------------------------------------
// ForcedAligner::Impl: loading
// ---------------------------------------------------------------------------

ForcedAligner::Impl::Impl(const ForcedAlignerConfig &config) : config_(config) {
  if (config_.model_dir.empty()) {
    throw std::invalid_argument("ForcedAligner: model_dir must not be empty");
  }
  if (config_.lexicon.empty()) {
    throw std::invalid_argument("ForcedAligner: lexicon must not be empty");
  }
  CheckModelFiles();

  meta_ = ParseFaModelMeta(ModelFile("meta.json"));
  LoadLdaTransform();
  LoadAcousticModel();
  LoadContextTree();
  lexicon_ = FaLexicon::Load(config_.lexicon);
  LoadPhoneTable();

  if (config_.debug) {
    PrintDebugInfo();
  }
}

// Renders a long list with only its head and tail kept, e.g. the phones list
// of meta.json.
static std::string FormatHeadTail(const std::vector<std::string> &items) {
  const size_t kNum = 8;
  size_t n = items.size();
  size_t head = std::min(n, kNum);
  size_t tail = n > kNum ? kNum : 0;

  std::ostringstream os;
  os << "[";
  for (size_t i = 0; i != head; ++i) {
    os << (i ? ", " : "") << items[i];
  }
  if (n > head + tail) {
    os << ", ... " << n - head - tail << " more ...";
    for (size_t i = n - tail; i != n; ++i) {
      os << ", " << items[i];
    }
  } else {
    for (size_t i = head; i != n; ++i) {
      os << ", " << items[i];
    }
  }
  os << "]";
  return os.str();
}

void ForcedAligner::Impl::PrintDebugInfo() const {
  const FaFeatureConfig &f = meta_.features;
  SHERPA_ONNX_LOGE("debug: meta.json:");
  SHERPA_ONNX_LOGE(
      "  features: sample_frequency=%f frame_shift=%f frame_length=%f "
      "snip_edges=%d dither=%f",
      f.sample_frequency, f.frame_shift, f.frame_length,
      static_cast<int>(f.snip_edges), f.dither);
  SHERPA_ONNX_LOGE(
      "  mfcc: use_energy=%d energy_floor=%f raw_energy=%d num_mel_bins=%d "
      "num_coefficients=%d cepstral_lifter=%f",
      static_cast<int>(f.use_energy), f.energy_floor,
      static_cast<int>(f.raw_energy), f.num_mel_bins, f.num_coefficients,
      f.cepstral_lifter);
  SHERPA_ONNX_LOGE(
      "  filter: low_frequency=%f high_frequency=%f preemphasis=%f",
      f.low_frequency, f.high_frequency, f.preemphasis_coefficient);
  SHERPA_ONNX_LOGE(
      "  pitch: use_pitch=%d min_f0=%f max_f0=%f delta_pitch=%f "
      "penalty_factor=%f",
      static_cast<int>(f.use_pitch), f.min_f0, f.max_f0, f.delta_pitch,
      f.penalty_factor);
  SHERPA_ONNX_LOGE(
      "  chain: uses_cmvn=%d uses_deltas=%d uses_splices=%d splice=%d/%d",
      static_cast<int>(f.uses_cmvn), static_cast<int>(meta_.uses_deltas),
      static_cast<int>(meta_.uses_splices), meta_.splice_left_context,
      meta_.splice_right_context);
  SHERPA_ONNX_LOGE(
      "  silence_probability=%f oov_phone=%s optional_silence_phone=%s",
      meta_.silence_probability, meta_.oov_phone.c_str(),
      meta_.optional_silence_phone.c_str());
  SHERPA_ONNX_LOGE("  phones (%d): %s", static_cast<int>(meta_.phones.size()),
                   FormatHeadTail(meta_.phones).c_str());
  SHERPA_ONNX_LOGE("debug: lexicon: %d keys",
                   static_cast<int>(lexicon_.entries.size()));
  SHERPA_ONNX_LOGE(
      "debug: phone table: %d phones, silence id %d, LDA transform: %d",
      static_cast<int>(phone2id_.size()), sil_id_, use_lda_ ? 1 : 0);
}

std::string ForcedAligner::Impl::ModelFile(const char *name) const {
  return config_.model_dir + "/" + name;
}

void ForcedAligner::Impl::CheckModelFiles() const {
  for (const char *name : {"meta.json", "final.alimdl", "tree"}) {
    std::string p = ModelFile(name);
    if (!FileExists(p)) {
      throw std::runtime_error("ForcedAligner: missing required file: '" + p +
                               "'");
    }
  }
  if (!FileExists(config_.lexicon)) {
    throw std::invalid_argument("ForcedAligner: lexicon does not exist: '" +
                                config_.lexicon + "'");
  }
}

void ForcedAligner::Impl::LoadLdaTransform() {
  std::string lda_path = ModelFile("lda.mat");
  use_lda_ = FileExists(lda_path);
  if (!use_lda_ && meta_.uses_splices) {
    throw std::runtime_error(
        "ForcedAligner: meta.json requests splice+LDA but lda.mat is "
        "missing: '" +
        lda_path + "'");
  }
  if (use_lda_) {
    // Loaded once here; Compute() only applies it.
    lda_ = std::make_unique<knf::FeatureTransform>(
        knf::FeatureTransform::Read(lda_path));
  }
}

void ForcedAligner::Impl::LoadAcousticModel() {
  // kaldi .mdl layout: TransitionModel followed by AmDiagGmm on one stream.
  std::string path = ModelFile("final.alimdl");
  try {
    bool binary = true;
    kaldiio::Input ki(path, &binary);
    trans_model_.Read(ki.Stream(), binary);
    am_.Read(ki.Stream(), binary);
  } catch (const std::exception &e) {
    throw std::runtime_error("ForcedAligner: failed to read '" + path +
                             "': " + e.what());
  }
}

void ForcedAligner::Impl::LoadContextTree() {
  std::string path = ModelFile("tree");
  try {
    bool binary = true;
    kaldiio::Input ki(path, &binary);
    tree_.Read(ki.Stream(), binary);
  } catch (const std::exception &e) {
    throw std::runtime_error("ForcedAligner: failed to read '" + path +
                             "': " + e.what());
  }
}

void ForcedAligner::Impl::LoadPhoneTable() {
  std::string phones_txt = ModelFile("phones.txt");
  if (FileExists(phones_txt)) {
    std::ifstream is = OpenInputFile(phones_txt);
    std::string line;
    while (std::getline(is, line)) {
      std::istringstream iss(line);
      std::string sym;
      int32_t id;
      if (iss >> sym >> id) {
        phone2id_[sym] = id;
        id2phone_[id] = sym;
      }
    }
  } else {
    // Older models ship no phones.txt. The ids then follow the kaldi
    // convention of these models: <eps>=0, sil=1, spn=2, then the lexical
    // phones of meta.json from 3.
    SHERPA_ONNX_LOGE(
        "phones.txt not found in '%s'; falling back to meta.json",
        config_.model_dir.c_str());
    phone2id_["sil"] = 1;
    phone2id_["spn"] = 2;
    id2phone_[1] = "sil";
    id2phone_[2] = "spn";
    for (size_t i = 0; i < meta_.phones.size(); ++i) {
      int32_t id = static_cast<int32_t>(i) + 3;
      phone2id_[meta_.phones[i]] = id;
      id2phone_[id] = meta_.phones[i];
    }

    // The convention holds only when meta.json lists the model's complete
    // phone set. MFA 2.0.x models (e.g. english_mfa v2.0.0) break it: they
    // ship no phones.txt and their model has more phones than meta.json
    // lists, so every phone would silently get a wrong pdf and the
    // alignment would be garbage. Refuse them; v2.2.1 and newer satisfy
    // the convention.
    const std::vector<int32_t> &model_phones = trans_model_.GetPhones();
    bool consistent = model_phones.size() == phone2id_.size();
    for (int32_t p : model_phones) {
      consistent = consistent && id2phone_.count(p) != 0;
    }
    if (!consistent) {
      throw std::runtime_error(
          "ForcedAligner: the model has no phones.txt, and the implicit "
          "phone table (" +
          std::to_string(phone2id_.size()) +
          " phones) does not match the " +
          std::to_string(model_phones.size()) +
          " phones of the model. MFA 2.0.x models (e.g. english_mfa "
          "v2.0.0) are known to be in this state and are not supported; "
          "use v2.2.1 or newer");
    }
  }
  if (phone2id_.empty()) {
    throw std::runtime_error(
        "ForcedAligner: no phone table (phones.txt missing and meta.json has "
        "no phones list)");
  }

  auto it = phone2id_.find(meta_.optional_silence_phone);
  if (it == phone2id_.end()) {
    throw std::runtime_error("ForcedAligner: optional silence phone '" +
                             meta_.optional_silence_phone +
                             "' is not in the phone table");
  }
  sil_id_ = it->second;
}

ForcedAligner::Impl::~Impl() = default;

std::unique_ptr<OfflineStream> ForcedAligner::Impl::CreateStream() const {
  return std::make_unique<OfflineStream>(meta_.features);
}

// ---------------------------------------------------------------------------
// ForcedAligner::Impl: alignment
// ---------------------------------------------------------------------------

namespace {

struct PhoneSegment {
  int32_t phone;
  float t0;
  float t1;
};

// Recovers word spans by matching each word's pronunciations against the
// phone segments of the alignment. Pronunciations of a word may differ in
// length, so the span must follow the pronunciation actually taken (the
// best-scoring parallel path), not the first one. Silence segments are
// unattributed gaps between words.
//
// Returns true when every word is matched; *spans[i] = (first, last) index
// into segs of the phone instances of words[i].
bool MatchWordSpans(
    const std::vector<int32_t> &word_ids, size_t word_idx,
    const std::unordered_map<int32_t, std::vector<std::vector<int32_t>>>
        &pronunciations,
    const std::vector<PhoneSegment> &segs, size_t seg_idx, int32_t sil_id,
    std::vector<std::pair<size_t, size_t>> *spans) {
  // Skip unattributed silence runs before this word.
  while (seg_idx < segs.size() && segs[seg_idx].phone == sil_id) {
    ++seg_idx;
  }

  if (word_idx == word_ids.size()) {
    // Trailing silence is allowed; anything else is a mismatch.
    for (size_t i = seg_idx; i < segs.size(); ++i) {
      if (segs[i].phone != sil_id) {
        return false;
      }
    }
    return true;
  }

  auto it = pronunciations.find(word_ids[word_idx]);
  if (it == pronunciations.end()) {
    return false;
  }

  for (const auto &pron : it->second) {
    if (seg_idx + pron.size() > segs.size()) {
      continue;
    }
    bool matches = true;
    for (size_t k = 0; k < pron.size(); ++k) {
      if (segs[seg_idx + k].phone != pron[k]) {
        matches = false;
        break;
      }
    }
    if (!matches) {
      continue;
    }
    spans->push_back({seg_idx, seg_idx + pron.size() - 1});
    if (MatchWordSpans(word_ids, word_idx + 1, pronunciations, segs,
                       seg_idx + pron.size(), sil_id, spans)) {
      return true;
    }
    spans->pop_back();
  }
  return false;
}

}  // namespace

khg::FloatMatrix ForcedAligner::Impl::ComputeFeatures(
    const OfflineStream *stream, bool debug) const {
  std::vector<float> frames = stream->GetFrames();
  int32_t feat_dim = stream->FeatureDim();
  if (frames.empty() || feat_dim <= 0) {
    return {};
  }
  int32_t num_frames = static_cast<int32_t>(frames.size() / feat_dim);

  // Zero-copy view of the stream features (MFCC -> [pitch] -> CMVN); the
  // trailing step below (splice + LDA, or deltas) is model-level and
  // materializes the matrix the decodable needs.
  Eigen::Map<const khg::FloatMatrix> raw_feats(frames.data(), num_frames,
                                               feat_dim);

  khg::FloatMatrix feats;
  if (use_lda_) {
    knf::SpliceOptions splice_opts;
    splice_opts.left_context = meta_.splice_left_context;
    splice_opts.right_context = meta_.splice_right_context;
    feats = lda_->Apply(knf::SpliceFrames(raw_feats, splice_opts));
  } else if (meta_.uses_deltas) {
    feats = knf::ComputeDeltas(knf::DeltaOptions(), raw_feats);
  } else {
    feats = raw_feats;
  }

  if (debug) {
    SHERPA_ONNX_LOGE("debug: features: %d frames x %d dim",
                     static_cast<int>(feats.rows()),
                     static_cast<int>(feats.cols()));
  }
  return feats;
}

ForcedAligner::Impl::ResolvedWords ForcedAligner::Impl::ResolveWords(
    const std::vector<std::string> &tokens, bool debug) const {
  // Pronunciations per token (context readings from multi-character keys);
  // the same token can resolve to different readings in different contexts,
  // so graph words are keyed by (token, pronunciations).
  auto resolved = ResolveFaPronunciations(tokens, lexicon_);

  struct DistinctWord {
    std::string word;
    std::vector<std::vector<int32_t>> prons;
  };
  std::vector<DistinctWord> distinct;

  ResolvedWords out;
  int32_t num_oov = 0;
  for (size_t wi = 0; wi < tokens.size(); ++wi) {
    const auto &w = tokens[wi];
    std::vector<std::vector<int32_t>> pron_ids;
    for (const auto &pron : resolved[wi]) {
      std::vector<int32_t> ids;
      ids.reserve(pron.size());
      for (const auto &p : pron) {
        auto pit = phone2id_.find(p);
        if (pit == phone2id_.end()) {
          throw std::runtime_error("ForcedAligner: phone '" + p + "' (word '" +
                                   w + "') is not in the phone table");
        }
        ids.push_back(pit->second);
      }
      if (!ids.empty() &&
          std::find(pron_ids.begin(), pron_ids.end(), ids) == pron_ids.end()) {
        pron_ids.push_back(std::move(ids));
      }
    }
    if (pron_ids.empty()) {
      ++num_oov;
      auto oit = phone2id_.find(meta_.oov_phone);
      if (oit == phone2id_.end()) {
        throw std::runtime_error("ForcedAligner: oov_phone '" +
                                 meta_.oov_phone +
                                 "' is not in the phone table");
      }
      pron_ids.push_back({oit->second});
    }

    int32_t wid = 0;
    for (size_t k = 0; k < distinct.size(); ++k) {
      if (distinct[k].word == w && distinct[k].prons == pron_ids) {
        wid = static_cast<int32_t>(k) + 1;
        break;
      }
    }
    if (wid == 0) {
      distinct.push_back({w, pron_ids});
      wid = static_cast<int32_t>(distinct.size());
    }
    out.word_ids.push_back(wid);
  }

  for (size_t k = 0; k < distinct.size(); ++k) {
    out.labels.push_back(distinct[k].word);
    out.pronunciations[static_cast<int32_t>(k) + 1] = distinct[k].prons;
  }

  if (num_oov == static_cast<int32_t>(tokens.size())) {
    throw std::runtime_error(
        "ForcedAligner: every word is out-of-vocabulary; the lexicon key "
        "shape does not match the text (e.g. hanzi text with a pinyin-keyed "
        "lexicon)");
  }
  if (num_oov > 0) {
    SHERPA_ONNX_LOGE("%d of %d words are out-of-vocabulary (mapped to '%s')",
                     num_oov, static_cast<int32_t>(tokens.size()),
                     meta_.oov_phone.c_str());
  }
  if (debug) {
    SHERPA_ONNX_LOGE("debug: words: %d distinct, %d OOV",
                     static_cast<int>(distinct.size()), num_oov);
  }
  return out;
}

std::vector<int32_t> ForcedAligner::Impl::DisambigSyms() const {
  // Disambiguation symbols (none in the usual phone tables).
  std::vector<int32_t> disambig;
  for (const auto &kv : phone2id_) {
    if (!kv.first.empty() && kv.first[0] == '#') {
      disambig.push_back(kv.second);
    }
  }
  return disambig;
}

fst::VectorFst<fst::StdArc> ForcedAligner::Impl::CompileAlignGraph(
    const ResolvedWords &resolved) const {
  // L.fst for the utterance's vocabulary: every pronunciation of a word is
  // a parallel path, so the acoustic scores pick the winning one.
  std::vector<FaFstWord> fst_words;
  std::vector<int32_t> transcript;
  std::unordered_map<int32_t, bool> emitted;
  for (int32_t wid : resolved.word_ids) {
    transcript.push_back(wid);
    if (emitted.count(wid)) {
      continue;
    }
    emitted[wid] = true;
    fst_words.push_back({wid, resolved.pronunciations.at(wid)});
  }
  fst::VectorFst<fst::StdArc> lex_fst =
      BuildFaLexiconFst(fst_words, sil_id_, meta_.silence_probability);

  // Compile the training graph (scales are added at alignment time).
  khg::TrainingGraphCompilerOptions gc_opts(0.0f, 0.0f);
  khg::TrainingGraphCompiler gc(
      trans_model_, tree_, new fst::VectorFst<fst::StdArc>(std::move(lex_fst)),
      DisambigSyms(), gc_opts);

  fst::VectorFst<fst::StdArc> align_fst;
  if (!gc.CompileGraphFromText(transcript, &align_fst)) {
    throw std::runtime_error(
        "ForcedAligner: failed to compile the alignment graph");
  }
  return align_fst;
}

std::vector<int32_t> ForcedAligner::Impl::Align(
    fst::VectorFst<fst::StdArc> *graph, const khg::FloatMatrix &feats,
    const ForcedAlignmentOptions &opts, bool debug,
    std::vector<int32_t> *aligned_words) const {
  khg::AddTransitionProbs(trans_model_, DisambigSyms(), opts.transition_scale,
                          opts.self_loop_scale, graph);
  khg::DecodableAmDiagGmmScaled decodable(am_, trans_model_, feats,
                                          opts.acoustic_scale);

  kaldi_decoder::AlignConfig align_config(opts.beam, opts.retry_beam,
                                          /*careful=*/false);
  int32_t num_done = 0;
  int32_t num_error = 0;
  int32_t num_retried = 0;
  double tot_like = 0;
  int64_t frame_count = 0;
  std::vector<int32_t> alignment;
  kaldi_decoder::AlignUtteranceWrapper(align_config, "utt", opts.acoustic_scale,
                                       graph, &decodable, &num_done, &num_error,
                                       &num_retried, &tot_like, &frame_count,
                                       &alignment, aligned_words);
  if (num_error != 0 || alignment.empty()) {
    throw std::runtime_error("ForcedAligner: alignment failed");
  }
  if (debug) {
    SHERPA_ONNX_LOGE("debug: alignment: log-like %f over %d frames", tot_like,
                     static_cast<int>(frame_count));
  }
  return alignment;
}

ForcedAlignmentResult ForcedAligner::Impl::BuildResult(
    const std::vector<int32_t> &alignment,
    const std::vector<int32_t> &aligned_words,
    const ResolvedWords &resolved) const {
  // Phone segments; adjacent instances of the same phone are not merged.
  std::vector<std::vector<int32_t>> split_alignment;
  khg::SplitToPhones(trans_model_, alignment, &split_alignment);

  float frame_shift = meta_.features.frame_shift / 1000.0f;
  std::vector<PhoneSegment> segs;
  segs.reserve(split_alignment.size());
  float t = 0.0f;
  for (const auto &seg : split_alignment) {
    float t1 = t + seg.size() * frame_shift;
    segs.push_back({trans_model_.TransitionIdToPhone(seg[0]), t, t1});
    t = t1;
  }

  ForcedAlignmentResult result;
  result.duration = segs.empty() ? 0.0f : segs.back().t1;

  AlignmentTier phone_tier;
  phone_tier.name = "phones";
  for (const auto &seg : segs) {
    auto it = id2phone_.find(seg.phone);
    std::string label = it != id2phone_.end() ? it->second : std::string("?");
    phone_tier.intervals.push_back({seg.t0, seg.t1, label});
  }
  result.tiers.push_back(std::move(phone_tier));

  // Word intervals follow the pronunciation path actually taken.
  std::vector<std::pair<size_t, size_t>> spans;
  if (!MatchWordSpans(aligned_words, 0, resolved.pronunciations, segs, 0,
                      sil_id_, &spans) ||
      spans.size() != aligned_words.size()) {
    SHERPA_ONNX_LOGE(
        "ForcedAligner: failed to recover word boundaries; returning an "
        "empty result");
    return {};
  }
  AlignmentTier word_tier;
  word_tier.name = "words";
  float word_t = 0.0f;
  for (size_t i = 0; i < spans.size(); ++i) {
    float t0 = segs[spans[i].first].t0;
    float t1 = segs[spans[i].second].t1;
    if (t0 > word_t) {
      // Silence gaps; Praat interval tiers must tile the full time range.
      word_tier.intervals.push_back({word_t, t0, ""});
    }
    word_tier.intervals.push_back(
        {t0, t1, resolved.labels[aligned_words[i] - 1]});
    word_t = t1;
  }
  if (result.duration > word_t) {
    word_tier.intervals.push_back({word_t, result.duration, ""});
  }
  result.tiers.push_back(std::move(word_tier));
  return result;
}

ForcedAlignmentResult ForcedAligner::Impl::Compute(
    const OfflineStream *stream, const std::string &text) const {
  if (!stream) {
    SHERPA_ONNX_LOGE("ForcedAligner: null stream; returning an empty result");
    return {};
  }
  if (text.empty()) {
    SHERPA_ONNX_LOGE("ForcedAligner: empty text; returning an empty result");
    return {};
  }

  const ForcedAlignmentOptions &d = config_.options;
  ForcedAlignmentOptions opts;
  opts.beam = stream->GetOptionFloat("beam", d.beam);
  opts.retry_beam = stream->GetOptionFloat("retry_beam", d.retry_beam);
  opts.transition_scale =
      stream->GetOptionFloat("transition_scale", d.transition_scale);
  opts.self_loop_scale =
      stream->GetOptionFloat("self_loop_scale", d.self_loop_scale);
  opts.acoustic_scale =
      stream->GetOptionFloat("acoustic_scale", d.acoustic_scale);
  bool debug = stream->GetOptionInt("debug", config_.debug);

  khg::FloatMatrix feats = ComputeFeatures(stream, debug);
  if (feats.rows() == 0) {
    SHERPA_ONNX_LOGE("ForcedAligner: no features; returning an empty result");
    return {};
  }

  std::vector<std::string> tokens = TokenizeFaText(text, lexicon_);
  if (tokens.empty()) {
    SHERPA_ONNX_LOGE(
        "ForcedAligner: no tokens after text normalization; returning an "
        "empty result");
    return {};
  }
  if (debug) {
    SHERPA_ONNX_LOGE("debug: tokens: %d", static_cast<int>(tokens.size()));
  }
  ResolvedWords words = ResolveWords(tokens, debug);

  fst::VectorFst<fst::StdArc> graph = CompileAlignGraph(words);
  std::vector<int32_t> aligned_words;
  std::vector<int32_t> alignment =
      Align(&graph, feats, opts, debug, &aligned_words);

  return BuildResult(alignment, aligned_words, words);
}

}  // namespace sherpa_onnx
