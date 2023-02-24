// sherpa-onnx/csrc/file-utils.h
//
// Copyright (c)  2022-2023  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_FILE_UTILS_H_
#define SHERPA_ONNX_CSRC_FILE_UTILS_H_

#include <fstream>
#include <string>

namespace sherpa_onnx {

/** Check whether a given path is a file or not
 *
 * @param filename Path to check.
 * @return Return true if the given path is a file; return false otherwise.
 */
bool FileExists(const std::string &filename);

/** Abort if the file does not exist.
 *
 * @param filename The file to check.
 */
void AssertFileExists(const std::string &filename);

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_FILE_UTILS_H_
