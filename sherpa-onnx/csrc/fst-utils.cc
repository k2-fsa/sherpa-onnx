// sherpa-onnx/csrc/fst-utils.cc
//
// Copyright (c)  2024  Xiaomi Corporation

#include "sherpa-onnx/csrc/fst-utils.h"

#include <algorithm>
#include <cmath>
#include <regex>
#include <vector>

#include "kaldifst/csrc/pre-determinize.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/utils.h"
namespace sherpa_onnx {

// This function is copied from kaldi.
//
// @param filename Path to a StdVectorFst or StdConstFst graph
// @return The caller should free the returned pointer using `delete` to
//         avoid memory leak.
fst::Fst<fst::StdArc> *ReadGraph(const std::string &filename) {
  // read decoding network FST
  std::ifstream is(filename, std::ios::binary);
  if (!is.good()) {
    SHERPA_ONNX_LOGE("Could not open decoding-graph FST %s", filename.c_str());
  }

  fst::FstHeader hdr;
  if (!hdr.Read(is, "<unknown>")) {
    SHERPA_ONNX_LOGE("Reading FST: error reading FST header.");
  }

  if (hdr.ArcType() != fst::StdArc::Type()) {
    SHERPA_ONNX_LOGE("FST with arc type %s not supported",
                     hdr.ArcType().c_str());
  }
  fst::FstReadOptions ropts("<unspecified>", &hdr);

  fst::Fst<fst::StdArc> *decode_fst = nullptr;

  if (hdr.FstType() == "vector") {
    decode_fst = fst::VectorFst<fst::StdArc>::Read(is, ropts);
  } else if (hdr.FstType() == "const") {
    decode_fst = fst::ConstFst<fst::StdArc>::Read(is, ropts);
  } else {
    SHERPA_ONNX_LOGE("Reading FST: unsupported FST type: %s",
                     hdr.FstType().c_str());
  }

  if (decode_fst == nullptr) {  // fst code will warn.
    SHERPA_ONNX_LOGE("Error reading FST (after reading header).");
    return nullptr;
  } else {
    return decode_fst;
  }
}

BuildFst::BuildFst(const std::string &text, const std::string &token_file_name,
                   const std::string &bpe_file_name, bool has_silence)
    : text(text),
      token_file_name(token_file_name),
      bpe_file_name(bpe_file_name),
      has_silence(has_silence) {
  if (!std::ifstream(token_file_name).good() ||
      !std::ifstream(bpe_file_name).good()) {
    SHERPA_ONNX_LOGE(
        "Need tokens.txt and bpe_en.vocab"
        "You can download the test data by: "
        "git clone https://github.com/pkufool/sherpa-test-data.git");
    exit(-1);
  }
  BuildLexicon();
}

fst::StdVectorFst *BuildFst::BuildH(bool tread_ilabel_zero_specially,
                                    bool update_olabel) {
  fst::StdVectorFst *H = new fst::StdVectorFst();
  BuildStandardCtcTopo(H);
  AddOne(H, tread_ilabel_zero_specially, update_olabel);
  return H;
}

void BuildFst::BuildL(fst::StdVectorFst *L) {  // ignore attach_symbol_table
  if (has_silence)
    MakeLexiconFstWithSilence(L);
  else
    MakeLexiconFstNoSilence(L);
}

fst::StdVectorFst *BuildFst::BuildHL() {
  fst::StdVectorFst *H = BuildH();
  fst::StdVectorFst L;
  BuildL(&L);

  if (has_silence)
    AddOne(&L, true, false);
  else
    AddOne(&L, false, false);

  ArcSort(H, fst::OLabelCompare<fst::StdArc>());
  ArcSort(&L, fst::OLabelCompare<fst::StdArc>());

  fst::StdVectorFst HL_;
  Compose(*H, L, &HL_);
  delete H;

  fst::StdVectorFst *HL = new fst::StdVectorFst();
  DeterminizeStar(HL_, HL);

  int32_t disambig0 = token2id["#0"] + 1;
  int32 max_disambig = max_disambig_id + 1;

  for (fst::StateIterator<fst::StdVectorFst> state(*HL); !state.Done();
       state.Next()) {
    for (fst::MutableArcIterator<fst::MutableFst<fst::StdArc>> arc(
             HL, state.Value());
         !arc.Done(); arc.Next()) {
      if (disambig0 <= arc.Value().ilabel &&
          arc.Value().ilabel <= max_disambig) {
        auto tmp = arc.Value();
        tmp.ilabel = 0;
        arc.SetValue(tmp);
      }
    }
  }
  return HL;
}

fst::StdVectorFst *BuildFst::BuildHLG(fst::StdVectorFst *G) {
  fst::StdVectorFst *H = BuildH();
  fst::StdVectorFst L;
  BuildL(&L);

  if (has_silence)
    AddOne(&L, true, false);
  else
    AddOne(&L, false, false);

  int32_t token_disambig0 = token2id["#0"] + 1;
  int32_t word_disambig0 = word2id["#0"];

  std::vector<int32_t> wdisambig_phones_int{token_disambig0};
  std::vector<int32_t> wdisambig_words_int{word_disambig0};
  AddSelfLoops(&L, wdisambig_phones_int, wdisambig_words_int);

  ArcSort(&L, fst::OLabelCompare<fst::StdArc>());
  ArcSort(G, fst::ILabelCompare<fst::StdArc>());

  fst::StdVectorFst LG;
  Compose(L, *G, &LG);

  fst::StdVectorFst LG_d;
  DeterminizeStar(LG, &LG_d);

  MinimizeEncoded(&LG_d);

  ArcSort(&LG_d, fst::ILabelCompare<fst::StdArc>());

  AddDisambigSelfLoops(H, token_disambig0, max_disambig_id + 1);

  ArcSort(H, fst::OLabelCompare<fst::StdArc>());

  fst::StdVectorFst HLG_;
  Compose(*H, LG_d, &HLG_);
  delete H;

  fst::StdVectorFst *HLG = new fst::StdVectorFst;
  DeterminizeStar(HLG_, HLG);

  int32_t max_disambig = max_disambig_id + 1;
  for (fst::StateIterator<fst::StdVectorFst> state(*HLG); !state.Done();
       state.Next()) {
    for (fst::MutableArcIterator<fst::MutableFst<fst::StdArc>> arc(
             HLG, state.Value());
         !arc.Done(); arc.Next()) {
      if (token_disambig0 <= arc.Value().ilabel &&
          arc.Value().ilabel <= max_disambig) {
        auto tmp = arc.Value();
        tmp.ilabel = 0;
        arc.SetValue(tmp);
      }
    }
  }
  return HLG;
}

// use an extra simple language model construction to generate G.fst
// https://github.com/alphacep/vosk-api/blob/master/src/language_model.cc
/*
fst::StdVectorFst *BuildFst::BuildHLG_local() {
  fst::StdVectorFst *H = BuildH();
  fst::StdVectorFst L;
  BuildL(&L);

  if (has_silence)
    AddOne(&L, true, false);
  else
    AddOne(&L, false, false);

  int32_t token_disambig0 = token2id["#0"] + 1;
  int32_t word_disambig0 = word2id["#0"];

  std::vector<int32_t> wdisambig_phones_int{token_disambig0};

  std::vector<int32_t> wdisambig_words_int{word_disambig0};

  AddSelfLoops(&L, wdisambig_phones_int, wdisambig_words_int);

  ArcSort(&L, fst::OLabelCompare<fst::StdArc>());

  LanguageModelOptions opts;
  opts.ngram_order = 2;
  opts.discount = 0.1;
  LanguageModelEstimator estimator(opts);
  std::vector<int32_t> wordid_for_G;
  for (auto word : prompt_word) wordid_for_G.push_back(word2id.at(word));

  estimator.AddCounts(wordid_for_G);
  std::vector<int32_t> sil_id_for_G({word2id.at("<UNK>")});
  estimator.AddCounts(sil_id_for_G);
  fst::StdVectorFst G;
  estimator.Estimate(&G);

  ArcSort(&G, fst::ILabelCompare<fst::StdArc>());

  fst::StdVectorFst LG;
  Compose(L, G, &LG);

  fst::StdVectorFst LG_d;
  DeterminizeStar(LG, &LG_d);

  MinimizeEncoded(&LG_d);

  ArcSort(&LG_d, fst::ILabelCompare<fst::StdArc>());

  AddDisambigSelfLoops(H, token_disambig0, max_disambig_id + 1);

  ArcSort(H, fst::OLabelCompare<fst::StdArc>());

  fst::StdVectorFst HLG_;
  Compose(*H, LG_d, &HLG_);
  delete H;

  fst::StdVectorFst *HLG = new fst::StdVectorFst;
  DeterminizeStar(HLG_, HLG);

  int32_t max_disambig = max_disambig_id + 1;
  for (fst::StateIterator<fst::StdVectorFst> state(*HLG); !state.Done();
       state.Next()) {
    for (fst::MutableArcIterator<fst::MutableFst<fst::StdArc>> arc(HLG,
                                                    state.Value());
         !arc.Done(); arc.Next()) {
      if (token_disambig0 <= arc.Value().ilabel &&
          arc.Value().ilabel <= max_disambig) {
        auto tmp = arc.Value();
        tmp.ilabel = 0;
        arc.SetValue(tmp);
      }
    }
  }
  return HLG;
}
*/

void BuildFst::GetWordsFromText(const std::string &text) {
  std::regex pattern(
      "(?:[a-zA-Z]+(?:\'[a-zA-Z]+|[a-zA-Z]*))");  // get words or words with
                                                  // single quote
  for (std::sregex_iterator it(text.begin(), text.end(), pattern), end_it;
       it != end_it; it++) {
    std::string tmp = it->str();
    transform(tmp.begin(), tmp.end(), tmp.begin(), ::toupper);
    prompt_word.push_back(std::move(tmp));
  }
}

void BuildFst::BuildLexicon() {
  GetWordsFromText(text);
  // build words.txt
  int32_t word_idx = 0;
  word2id.insert({"<eps>", word_idx++});
  word2id.insert({"!SIL", word_idx++});
  word2id.insert({"<SPOKNE_NOISE>", word_idx++});
  word2id.insert({"<UNK>", word_idx++});
  std::string words_for_bpe;
  for (const auto &word : prompt_word) {
    if (word2id.find(word) == word2id.end()) {  // no repear word
      word2id.insert({word, word_idx++});
      words_for_bpe += word + "\n";
      unique_prompt_word.push_back(word);
    }
  }
  word2id.insert({"#0", word_idx++});
  word2id.insert({"<s>", word_idx++});
  word2id.insert({"</s>", word_idx++});

  token2id = SymbolTable(token_file_name);
  GetMaxTokenID();

  auto bpe_processor =
      std::make_unique<ssentencepiece::Ssentencepiece>(bpe_file_name);

  std::istringstream iss(words_for_bpe);

  std::vector<std::vector<int32_t>> ids;
  std::vector<float> scores;

  bool r =
      EncodeHotwords(iss, "bpe", token2id, bpe_processor.get(), &ids, &scores);
  assert(ids.size() == unique_prompt_word.size());
  if (!r) {
    // I don't know how to deal error
  } else {
    for (int i = 0; i < ids.size(); i++)
      word2tokenid.insert({unique_prompt_word[i], ids[i]});
  }
  word2tokenid.insert(
      {"<UNK>", {token2id["<unk>"]}});  // same as lexicon.txt, maybe useless
}

void BuildFst::GetMaxTokenID() {
  int32_t token_num = token2id.NumSymbols();
  std::regex pattern("^#\\d+$");
  for (int32_t i = 0; i < token_num; i++) {
    assert(token2id.Contains(i));
    if (std::regex_match(token2id[i], pattern))
      max_disambig_id = i;
    else
      max_token_id = i;
  }
}

void BuildFst::AddOne(fst::StdVectorFst *fst, bool treat_ilabel_zero_specially,
                      bool update_olabel) const {
  for (fst::StateIterator<fst::StdVectorFst> state(*fst); !state.Done();
       state.Next()) {
    for (fst::MutableArcIterator<fst::MutableFst<fst::StdArc>> arc(
             fst, state.Value());
         !arc.Done(); arc.Next()) {
      auto tmp = arc.Value();
      if (treat_ilabel_zero_specially == false || tmp.ilabel != 0)
        tmp.ilabel += 1;
      if (update_olabel && tmp.olabel != 0) tmp.olabel += 1;
      arc.SetValue(tmp);
    }
  }
}

void BuildFst::BuildStandardCtcTopo(fst::StdVectorFst *fst) const {
  int32_t num_states = max_token_id + 1;
  for (int32_t i = 0; i < num_states; i++) {
    auto s = fst->AddState();
    fst->SetFinal(s, 0);
  }
  fst->SetStart(0);

  for (int32_t i = 0; i < num_states; i++)
    for (int32_t k = 0; k < num_states; k++)
      fst->AddArc(i, fst::StdArc(k, i != k ? k : 0, 0, k));
}

void BuildFst::MakeLexiconFstNoSilence(fst::StdVectorFst *fst) const {
  auto start_state = fst->AddState();
  fst->SetStart(start_state);
  fst->SetFinal(start_state, 0);

  for (auto iter = word2tokenid.begin(); iter != word2tokenid.end(); iter++) {
    auto word = iter->first;
    auto phoneseq = iter->second;
    float pron_cost = 0;
    auto cur_state = start_state;
    for (int32_t i = 0; i < phoneseq.size() - 1; ++i) {
      auto next_state = fst->AddState();
      fst->AddArc(cur_state,
                  fst::StdArc(phoneseq[i], i == 0 ? word2id.at(word) : 0,
                              i == 0 ? pron_cost : 0, next_state));
      cur_state = std::move(next_state);
    }

    int32_t i = phoneseq.size() - 1;
    fst->AddArc(cur_state, fst::StdArc(i >= 0 ? phoneseq[i] : 0,
                                       i == 0 ? word2id.at(word) : 0,
                                       i == 0 ? pron_cost : 0, start_state));
  }
}

void BuildFst::MakeLexiconFstWithSilence(fst::StdVectorFst *fst, float sil_prob,
                                         std::string sil_phone) const {
  float sil_cost = -1 * logf(sil_prob);
  float no_sil_cost = -1 * logf(1.f - sil_prob);
  auto start_state = fst->AddState();
  auto loop_state = fst->AddState();
  auto sil_state = fst->AddState();

  fst->SetStart(start_state);
  fst->SetFinal(loop_state, 0);
  fst->AddArc(start_state, fst::StdArc(0, 0, no_sil_cost, loop_state));
  fst->AddArc(start_state, fst::StdArc(0, 0, sil_cost, sil_state));
  fst->AddArc(sil_state, fst::StdArc(token2id[sil_phone], 0, 0, loop_state));

  for (auto iter = word2tokenid.begin(); iter != word2tokenid.end(); iter++) {
    auto word = iter->first;
    auto phoneseq = iter->second;
    float pron_cost = 0;
    auto cur_state = start_state;

    for (int32_t i = 0; i < phoneseq.size() - 1; ++i) {
      auto next_state = fst->AddState();
      fst->AddArc(cur_state,
                  fst::StdArc(phoneseq[i], i == 0 ? word2id.at(word) : 0,
                              i == 0 ? pron_cost : 0, next_state));
      cur_state = std::move(next_state);
    }

    int32_t i = phoneseq.size() - 1;
    fst->AddArc(
        cur_state,
        fst::StdArc(i >= 0 ? phoneseq[i] : 0, i <= 0 ? word2id.at(word) : 0,
                    no_sil_cost + i <= 0 ? pron_cost : 0, loop_state));

    fst->AddArc(
        cur_state,
        fst::StdArc(i >= 0 ? phoneseq[i] : 0, i <= 0 ? word2id.at(word) : 0,
                    sil_cost + i <= 0 ? pron_cost : 0, sil_state));
  }
}

void BuildFst::AddDisambigSelfLoops(fst::StdVectorFst *fst, int32_t start,
                                    int32_t end) const {
  for (fst::StateIterator<fst::StdVectorFst> state(*fst); !state.Done();
       state.Next()) {
    for (int i = start; i < end + 1; i++) {
      fst->AddArc(state.Value(), fst::StdArc(i, i, 0, state.Value()));
    }
  }
}

}  // namespace sherpa_onnx