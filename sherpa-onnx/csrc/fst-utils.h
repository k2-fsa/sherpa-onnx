// sherpa-onnx/csrc/fst-utils.h
//
// Copyright (c)  2024  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_FST_UTILS_H_
#define SHERPA_ONNX_CSRC_FST_UTILS_H_

#include <string>

#include "kaldifst/csrc/fstext-utils.h"
#include "sherpa-onnx/csrc/symbol-table.h"
namespace sherpa_onnx {

fst::Fst<fst::StdArc> *ReadGraph(const std::string &filename);

class BuildFst {  // ignore input_symbols and output_symbols in FST
 public:
  explicit BuildFst(const std::string &text, const std::string &tokens,
                    const std::string &bpe, bool has_silence = false);
  BuildFst() = default;
  BuildFst(const BuildFst &) = delete;
  BuildFst &operator=(const BuildFst &) = delete;

  // remeber to delete the returned pointer after use or use it by a unique_ptr
  fst::StdVectorFst *BuildH(bool tread_ilabel_zero_specially = false,
                            bool update_olabel = true);
  fst::StdVectorFst *BuildHL();
  fst::StdVectorFst *BuildHLG(fst::StdVectorFst *G);

  // use an extra simple language model construction to generate G.fst
  // https://github.com/alphacep/vosk-api/blob/master/src/language_model.cc
  // fst::StdVectorFst * BuildHLG_local();

 private:
  bool has_silence = false;  // no SIL in token.txt, I'm not sure what happened
                             // about has_silence=true
  int32_t max_token_id = 0;
  int32_t max_disambig_id = 0;

  std::string text;
  std::string token_file_name;
  std::string bpe_file_name;
  std::string word_for_bpe;

  SymbolTable token2id;

  void BuildL(fst::StdVectorFst *L);
  std::vector<std::string>
      prompt_word;  // repeated word is OK, also use for simple G.fst in BuildHLG_local()
  std::vector<std::string>
      unique_prompt_word;  // no repeat word, elements are treated as index
  std::unordered_map<std::string, int32_t> word2id;
  std::unordered_map<std::string, std::vector<int32_t>>
      word2tokenid;  // for example, {helps, {22, 25, 21, 3}}
  void GetWordsFromText(const std::string &text);
  void BuildLexicon(); // generate lexicon by bpe
  void GetMaxTokenID();
  void AddOne(fst::StdVectorFst *fst, bool treat_ilabel_zero_specially,
              bool update_olabel) const;
  void BuildStandardCtcTopo(fst::StdVectorFst *fst) const;
  void MakeLexiconFstNoSilence(fst::StdVectorFst *fst) const;

  void MakeLexiconFstWithSilence(
      fst::StdVectorFst *fst, float sil_prob = 0.5f,
      std::string sil_phone =
          "SIL") const;  // actually token.txt doesn't contain "SIL"

  void AddDisambigSelfLoops(fst::StdVectorFst *fst, int32_t start, int32_t end) const;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_FST_UTILS_H_
