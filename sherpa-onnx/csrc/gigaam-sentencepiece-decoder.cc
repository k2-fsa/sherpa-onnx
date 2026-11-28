// sherpa-onnx/csrc/gigaam-sentencepiece-decoder.cc
//
// Copyright (c)  2026  sherpa-onnx contributors

#include "sherpa-onnx/csrc/gigaam-sentencepiece-decoder.h"

#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/file-utils.h"

namespace sherpa_onnx {

namespace {

// The field numbers below follow SentencePiece's sentencepiece_model.proto.
// Read only the ModelProto fields needed to decode these release artifacts.
class ProtoReader {
 public:
  explicit ProtoReader(std::string_view bytes) : bytes_(bytes) {}

  bool Next() {
    if (bytes_.empty()) {
      return false;
    }
    const uint64_t tag = Varint();
    number = tag >> 3;
    wire = tag & 7;
    integer = 0;
    payload = {};
    if (number == 0) {
      throw std::runtime_error("Invalid SentencePiece field number");
    }
    switch (wire) {
      case 0:
        integer = Varint();
        break;
      case 1:
        Take(8);
        break;
      case 2:
        payload = Take(Varint());
        break;
      case 5:
        Take(4);
        break;
      default:
        throw std::runtime_error("Invalid SentencePiece field encoding");
    }
    return true;
  }

  uint64_t number = 0;
  uint64_t wire = 0;
  uint64_t integer = 0;
  std::string_view payload;

 private:
  uint64_t Varint() {
    uint64_t value = 0;
    for (int shift = 0; shift < 64 && !bytes_.empty(); shift += 7) {
      const auto byte = static_cast<uint8_t>(bytes_.front());
      bytes_.remove_prefix(1);
      if (shift == 63 && byte > 1) {
        break;
      }
      value |= static_cast<uint64_t>(byte & 127) << shift;
      if (byte < 128) {
        return value;
      }
    }
    throw std::runtime_error("Invalid or truncated SentencePiece varint");
  }

  std::string_view Take(uint64_t size) {
    if (size > bytes_.size()) {
      throw std::runtime_error("Truncated SentencePiece field");
    }
    auto data = bytes_.substr(0, size);
    bytes_.remove_prefix(size);
    return data;
  }

  std::string_view bytes_;
};

constexpr std::string_view kSpace = "\xe2\x96\x81";

}  // namespace

GigaAMSentencePieceDecoder::GigaAMSentencePieceDecoder(
    const std::string &filename) {
  const auto bytes = ReadFile(filename);
  if (bytes.empty()) {
    throw std::runtime_error("Empty GigaAM SentencePiece model");
  }

  ProtoReader model(std::string_view(bytes.data(), bytes.size()));
  while (model.Next()) {
    if (model.wire != 2) {
      continue;
    }
    switch (model.number) {
      case 1:
        tokens_.push_back(ParsePiece(model.payload));
        break;
      case 2:
        ReadTrainer(model.payload);
        break;
      case 3:
        ReadNormalizer(model.payload);
        break;
      case 5:
        if (!model.payload.empty()) {
          throw std::runtime_error(
              "Unsupported GigaAM SentencePiece denormalizer");
        }
        break;
    }
  }

  if (tokens_.empty()) {
    throw std::runtime_error("No pieces in GigaAM SentencePiece model");
  }
}

GigaAMSentencePieceDecoder::Token GigaAMSentencePieceDecoder::ParsePiece(
    std::string_view payload) {
  Token token;
  ProtoReader field(payload);
  while (field.Next()) {
    if (field.number == 1 && field.wire == 2) {
      token.piece = field.payload;
    } else if (field.number == 3 && field.wire == 0) {
      token.type = field.integer;
    }
  }
  if (token.piece.empty() || token.type < 1 || token.type > 5) {
    throw std::runtime_error("Unsupported GigaAM SentencePiece token");
  }
  return token;
}

void GigaAMSentencePieceDecoder::ReadTrainer(std::string_view payload) {
  ProtoReader field(payload);
  while (field.Next()) {
    if (field.number == 44 && field.wire == 2) {
      unknown_surface_ = field.payload;
    }
  }
}

void GigaAMSentencePieceDecoder::ReadNormalizer(std::string_view payload) {
  ProtoReader field(payload);
  while (field.Next()) {
    if (field.wire != 0) {
      continue;
    }
    if (field.number == 3) {
      add_dummy_prefix_ = field.integer != 0;
    } else if (field.number == 4) {
      remove_extra_whitespaces_ = field.integer != 0;
    }
  }
}

const GigaAMSentencePieceDecoder::Token &GigaAMSentencePieceDecoder::GetToken(
    int64_t id) const {
  if (id < 0 || static_cast<uint64_t>(id) >= tokens_.size()) {
    throw std::runtime_error(
        "GigaAM token ID is outside the selected SentencePiece vocabulary");
  }
  return tokens_[id];
}

const std::string &GigaAMSentencePieceDecoder::Piece(int64_t id) const {
  return GetToken(id).piece;
}

std::string GigaAMSentencePieceDecoder::Decode(
    const std::vector<int64_t> &ids) const {
  std::string text;
  bool at_start = true;
  for (const auto id : ids) {
    const auto &token = GetToken(id);
    if (token.type == 3) {  // CONTROL pieces have no surface.
      continue;
    }
    if (token.type == 2) {  // UNKNOWN uses the model's configured surface.
      text += unknown_surface_;
    } else {
      std::string_view piece = token.piece;
      if (at_start && (add_dummy_prefix_ || remove_extra_whitespaces_) &&
          piece.substr(0, kSpace.size()) == kSpace) {
        piece.remove_prefix(kSpace.size());
        at_start = remove_extra_whitespaces_;
      }
      size_t pos = piece.find(kSpace);
      while (pos != std::string_view::npos) {
        text += piece.substr(0, pos);
        text += ' ';
        piece.remove_prefix(pos + kSpace.size());
        pos = piece.find(kSpace);
      }
      text += piece;
    }
    at_start = at_start && text.empty();
  }
  return text;
}

}  // namespace sherpa_onnx
