// sherpa-onnx/csrc/qwen-asr-tokenizer.h
//
// Copyright (c)  2026  zengyw
#ifndef SHERPA_ONNX_CSRC_QWEN_ASR_TOKENIZER_H_
#define SHERPA_ONNX_CSRC_QWEN_ASR_TOKENIZER_H_

#include <array>
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#if __ANDROID_API__ >= 9
#include <android/asset_manager.h>
#endif

#if __OHOS__
struct NativeResourceManager;
#endif

namespace sherpa_onnx {

class QwenAsrTokenizer {
 public:
  explicit QwenAsrTokenizer(const std::string &tokenizer_dir);

#if __ANDROID_API__ >= 9
  QwenAsrTokenizer(AAssetManager *mgr, const std::string &tokenizer_dir);
#endif

#if __OHOS__
  QwenAsrTokenizer(NativeResourceManager *mgr,
                   const std::string &tokenizer_dir);
#endif

  std::vector<int64_t> Encode(const std::string &text);
  std::string Decode(const std::vector<int64_t> &token_ids);
  std::string GetTokenStringStreaming(int64_t token_id,
                                      std::string *pending_bytes) const;

  int64_t GetEosTokenId() const { return eos_token_id_; }
  int64_t GetPadTokenId() const { return pad_token_id_; }
  int64_t GetImEndTokenId() const { return im_end_token_id_; }

 private:
  void Init(const std::string &tokenizer_dir);
  void InitFromContents(const std::string &vocab_content,
                        const std::string &merges_content,
                        const std::string &config_content,
                        const std::string &tokenizer_dir);

#if __ANDROID_API__ >= 9
  void Init(AAssetManager *mgr, const std::string &tokenizer_dir);
#endif

#if __OHOS__
  void Init(NativeResourceManager *mgr, const std::string &tokenizer_dir);
#endif

  int64_t eos_token_id_ = -1;
  int64_t pad_token_id_ = -1;
  int64_t im_end_token_id_ = -1;
  int64_t unk_token_id_ = -1;

  std::unordered_map<std::string, int32_t> token2id_;
  std::vector<std::string> id2token_;

  std::vector<std::pair<std::string, int32_t>> special_tokens_;

  std::array<std::string, 256> byte_to_unicode_;
  std::unordered_map<std::string, int32_t> merges_rank_;
  mutable std::unordered_map<std::string, std::vector<std::string>> bpe_cache_;
  mutable std::mutex bpe_cache_mutex_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_QWEN_ASR_TOKENIZER_H_
