// sherpa-onnx/csrc/offline-tts-zipvoice-frontend.h
//
// Copyright (c)  2025  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_OFFLINE_TTS_ZIPVOICE_FRONTEND_H_
#define SHERPA_ONNX_CSRC_OFFLINE_TTS_ZIPVOICE_FRONTEND_H_
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "cppinyin/csrc/cppinyin.h"
#include "sherpa-onnx/csrc/offline-tts-frontend.h"
#include "sherpa-onnx/csrc/offline-tts-zipvoice-model-meta-data.h"

namespace sherpa_onnx {

class OfflineTtsZipvoiceFrontend : public OfflineTtsFrontend {
 public:
  OfflineTtsZipvoiceFrontend(const std::string &tokens,
                             const std::string &data_dir,
                             const std::string &pinyin_dict,
                             const OfflineTtsZipvoiceModelMetaData &meta_data,
                             bool debug = false);

  template <typename Manager>
  OfflineTtsZipvoiceFrontend(Manager *mgr, const std::string &tokens,
                             const std::string &data_dir,
                             const std::string &pinyin_dict,
                             const OfflineTtsZipvoiceModelMetaData &meta_data,
                             bool debug = false);

  /** Convert a string to token IDs.
   *
   * @param text The input text.
   *             Example 1: "This is the first sample sentence; this is the
   *             second one." Example 2: "这是第一句。这是第二句。"
   * @param voice Optional. It is for espeak-ng.
   *
   * @return Return a vector-of-vector of token IDs. Each subvector contains
   *         a sentence that can be processed independently.
   *         If a frontend does not support splitting the text into
   * sentences, the resulting vector contains only one subvector.
   */
  std::vector<TokenIDs> ConvertTextToTokenIds(
      const std::string &text, const std::string &voice = "") const override;

 private:
  bool debug_ = false;
  std::unordered_map<std::string, int32_t> token2id_;
  const std::unordered_map<std::string, std::string> punct_map_ = {
      {"，", ","}, {"。", "."}, {"！", "!"},  {"？", "?"},     {"；", ";"},
      {"：", ":"}, {"、", ","}, {"‘", "'"},   {"“", "\""},     {"”", "\""},
      {"’", "'"},  {"⋯", "…"},  {"···", "…"}, {"・・・", "…"}, {"...", "…"}};
  OfflineTtsZipvoiceModelMetaData meta_data_;
  std::unique_ptr<cppinyin::PinyinEncoder> pinyin_encoder_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_TTS_ZIPVOICE_FRONTEND_H_
