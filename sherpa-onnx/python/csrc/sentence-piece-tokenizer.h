// sherpa-onnx/python/csrc/sentence-piece-tokenizer.h
//
// Copyright (c)  2026  Xiaomi Corporation

#ifndef SHERPA_ONNX_PYTHON_CSRC_SENTENCE_PIECE_TOKENIZER_H_
#define SHERPA_ONNX_PYTHON_CSRC_SENTENCE_PIECE_TOKENIZER_H_

#include "sherpa-onnx/python/csrc/sherpa-onnx.h"

namespace sherpa_onnx {

void PybindSentencePieceTokenizer(py::module *m);

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_PYTHON_CSRC_SENTENCE_PIECE_TOKENIZER_H_
