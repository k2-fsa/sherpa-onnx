// sherpa-onnx/csrc/fst-utils.h
//
// Copyright (c)  2024  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_FST_UTILS_H_
#define SHERPA_ONNX_CSRC_FST_UTILS_H_

#include <memory>
#include <string>
#include <vector>

#include "fst/fstlib.h"

namespace sherpa_onnx {

fst::Fst<fst::StdArc> *ReadGraph(const std::string &filename);

std::vector<std::unique_ptr<fst::StdConstFst>> ReadFstsFromFar(
    const std::vector<char> &buffer);

}

#endif  // SHERPA_ONNX_CSRC_FST_UTILS_H_
