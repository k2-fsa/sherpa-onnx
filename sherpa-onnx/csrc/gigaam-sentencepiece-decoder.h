// sherpa-onnx/csrc/gigaam-sentencepiece-decoder.h
//
// Copyright (c)  2026  sherpa-onnx contributors

#ifndef SHERPA_ONNX_CSRC_GIGAAM_SENTENCEPIECE_DECODER_H_
#define SHERPA_ONNX_CSRC_GIGAAM_SENTENCEPIECE_DECODER_H_

#include <stdint.h>

#include <string>
#include <vector>

namespace sherpa_onnx {

// Decodes the released GigaAM SentencePiece models without encoding text.
class GigaAMSentencePieceDecoder {
 public:
  explicit GigaAMSentencePieceDecoder(const std::string &filename);
  const std::string &Piece(int64_t id) const;
  std::string Decode(const std::vector<int64_t> &ids) const;

 private:
  struct Token {
    std::string piece;
    uint64_t type = 1;  // Default ModelProto.SentencePiece type is NORMAL.
  };

  std::vector<Token> tokens_;
  std::string unknown_surface_ = " \xe2\x81\x87 ";
  bool add_dummy_prefix_ = true;
  bool remove_extra_whitespaces_ = true;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_GIGAAM_SENTENCEPIECE_DECODER_H_
