// sherpa-onnx/csrc/funasr-nano-tokenizer.h
//
// Copyright (c)  2025  zengyw
//
// A self-contained Qwen3 ByteLevel-BPE tokenizer implementation.
// - No dependency on tokenizers-cpp / HF tokenizers
// - Loads vocab.json + merges.txt + tokenizer.json(added_tokens)
// - Supports AddedTokens via Trie longest-match
// - ByteLevel bytes_to_unicode encode/decode

#ifndef SHERPA_ONNX_CSRC_FUNASR_NANO_TOKENIZER_H_
#define SHERPA_ONNX_CSRC_FUNASR_NANO_TOKENIZER_H_

#include <cstdint>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#if __ANDROID_API__ >= 9
#include <android/asset_manager.h>
#endif

#if __OHOS__
struct NativeResourceManager;
#endif

namespace sherpa_onnx {

class FunASRNanoTokenizer {
 public:
  explicit FunASRNanoTokenizer(const std::string &tokenizer_dir);

#if __ANDROID_API__ >= 9
  FunASRNanoTokenizer(AAssetManager *mgr, const std::string &tokenizer_dir);
#endif

#if __OHOS__
  FunASRNanoTokenizer(NativeResourceManager *mgr,
                      const std::string &tokenizer_dir);
#endif

  std::vector<int64_t> Encode(const std::string &text);
  std::string Decode(const std::vector<int64_t> &token_ids);
  std::string GetTokenStringStreaming(int64_t token_id,
                                      std::string *pending_bytes) const;

  int64_t GetEosTokenId() const { return eos_token_id_; }
  int64_t GetPadTokenId() const { return pad_token_id_; }
  int64_t GetImEndTokenId() const { return im_end_token_id_; }

  // Public structures for helper functions
  struct AddedToken {
    std::string content;
    int32_t id = -1;
    bool single_word = false;
    bool lstrip = false;
    bool rstrip = false;
    bool normalized = false;
    bool special = false;
  };

  struct TrieNode {
    std::unordered_map<uint8_t, int32_t> next;
    int32_t token_index = -1;  // index in added_tokens_ if terminal
  };

 private:
  void Init(const std::string &tokenizer_dir);

#if __ANDROID_API__ >= 9
  void Init(AAssetManager *mgr, const std::string &tokenizer_dir);
#endif

#if __OHOS__
  void Init(NativeResourceManager *mgr, const std::string &tokenizer_dir);
#endif

  void FinalizeSpecialIds();

 private:
  // Special ids
  int64_t eos_token_id_ = -1;
  int64_t pad_token_id_ = -1;
  int64_t im_end_token_id_ = -1;

  std::unordered_set<int32_t> special_ids_;

  // Vocab: token <-> id
  std::unordered_map<std::string, int32_t> token2id_;
  std::vector<std::string> id2token_;

  // merges ranks: "left\tright" -> rank
  std::unordered_map<std::string, int32_t> merges_rank_;

  // BPE cache: bytelevel_word -> list of merged tokens
  std::unordered_map<std::string, std::vector<std::string>> bpe_cache_;

  // bytes_to_unicode mapping (ByteLevel)
  std::string byte_to_unicode_[256];
  std::unordered_map<std::string, uint8_t> unicode_to_byte_;

  // AddedTokens
  std::vector<AddedToken> added_tokens_;
  std::vector<TrieNode> trie_;
  std::unordered_set<std::string> added_token_contents_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_FUNASR_NANO_TOKENIZER_H_
