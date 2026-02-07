// sherpa-onnx/csrc/file-utils.h
//
// Copyright (c)  2022-2023  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_FILE_UTILS_H_
#define SHERPA_ONNX_CSRC_FILE_UTILS_H_

#include <fstream>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#if __OHOS__
#include "rawfile/raw_file_manager.h"
#endif

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

std::vector<char> ReadFile(const std::string &filename);

#if __ANDROID_API__ >= 9
std::vector<char> ReadFile(AAssetManager *mgr, const std::string &filename);
#endif

#if __OHOS__
std::vector<char> ReadFile(NativeResourceManager *mgr,
                           const std::string &filename);
#endif

std::string ResolveAbsolutePath(const std::string &path);

// Load JSON from file.
nlohmann::json LoadJsonFromFile(const std::string &path);

#if __ANDROID_API__ >= 9
// Load JSON from Android asset manager
nlohmann::json LoadJsonFromFile(AAssetManager *mgr, const std::string &path);
#endif

#if __OHOS__
// Load JSON from OHOS resource manager
nlohmann::json LoadJsonFromFile(NativeResourceManager *mgr,
                                const std::string &path);
#endif

// Load JSON from buffer (for Android/OHOS)
nlohmann::json LoadJsonFromBuffer(const std::vector<char> &buf);

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_FILE_UTILS_H_
