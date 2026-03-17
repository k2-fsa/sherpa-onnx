// sherpa-onnx/csrc/file-utils.cc
//
// Copyright (c)  2022-2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/file-utils.h"

#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/stat.h>
#include <unistd.h>
#include <limits.h>
#include <stdlib.h>
#endif

#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {
std::wstring ToWideString(const std::string &s);
std::string ToString(const std::wstring &s);

bool FileExists(const std::string &filename) {
#ifdef _WIN32
    DWORD attributes = GetFileAttributesW(ToWideString(filename).c_str());
    
    return attributes != INVALID_FILE_ATTRIBUTES && !(attributes & FILE_ATTRIBUTE_DIRECTORY);
#else
    struct stat file_stat;
    return stat(filename.c_str(), &file_stat) == 0 && S_ISREG(file_stat.st_mode);
#endif
}

void AssertFileExists(const std::string &filename) {
  if (!FileExists(filename)) {
    SHERPA_ONNX_LOGE("filename '%s' does not exist", filename.c_str());
    SHERPA_ONNX_EXIT(-1);
  }
}

std::vector<char> ReadFile(const std::string &filename) {
  if (filename.empty()) {
    return {};
  }
  try {
#ifdef _WIN32     
    HANDLE hFile = CreateFileW(
        ToWideString(filename).c_str(),
        GENERIC_READ,
        FILE_SHARE_READ,
        nullptr,
        OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL,
        nullptr
    );

    if (hFile == INVALID_HANDLE_VALUE) {
      return {};
    }

    std::unique_ptr<void, decltype(&CloseHandle)> file_guard(
      hFile, CloseHandle);

    LARGE_INTEGER file_size;
    if (!GetFileSizeEx(hFile, &file_size) || file_size.QuadPart > SIZE_MAX) {
      return {};
    }

    std::vector<char> buffer(static_cast<size_t>(file_size.QuadPart));

    DWORD bytes_read = 0;
    bool read_success = ::ReadFile(
      hFile, 
      buffer.data(), 
      static_cast<DWORD>(buffer.size()), 
      &bytes_read, 
      nullptr
    );
    if (!read_success || bytes_read != buffer.size()) {
      return {};
    }
    
    return buffer;
#else
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
      return {};
    }

    std::streamsize size = file.tellg();
    if (size < 0) {
      return {};
    }
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(static_cast<size_t>(size));
    if (!file.read(buffer.data(), size)) {
      return {};
    }

    return buffer;
#endif
  } catch (const std::exception&) {
    return {};
  }
}

#if __ANDROID_API__ >= 9
std::vector<char> ReadFile(AAssetManager *mgr, const std::string &filename) {
  if (!filename.empty() && filename[0] == '/') {
    SHERPA_ONNX_LOGE(
        "You are using an absolute path '%s', but assetManager is NOT set to "
        "null.",
        filename.c_str());

    SHERPA_ONNX_LOGE(
        "Please set assetManager to null when you load model files from the SD "
        "card");

    SHERPA_ONNX_LOGE(
        "See also https://github.com/k2-fsa/sherpa-onnx/issues/2562");
  }

  AAsset *asset = AAssetManager_open(mgr, filename.c_str(), AASSET_MODE_BUFFER);
  if (!asset) {
    __android_log_print(ANDROID_LOG_FATAL, "sherpa-onnx",
                        "Read binary file: Load '%s' failed", filename.c_str());
    exit(-1);
  }

  auto p = reinterpret_cast<const char *>(AAsset_getBuffer(asset));
  size_t asset_length = AAsset_getLength(asset);

  std::vector<char> buffer(p, p + asset_length);
  AAsset_close(asset);

  return buffer;
}
#endif

#if __OHOS__
std::vector<char> ReadFile(NativeResourceManager *mgr,
                           const std::string &filename) {
  std::unique_ptr<RawFile, decltype(&OH_ResourceManager_CloseRawFile)> fp(
      OH_ResourceManager_OpenRawFile(mgr, filename.c_str()),
      OH_ResourceManager_CloseRawFile);

  if (!fp) {
    std::ostringstream os;
    os << "Read file '" << filename << "' failed.";
    SHERPA_ONNX_LOGE("%s", os.str().c_str());
    return {};
  }

  auto len = static_cast<int32_t>(OH_ResourceManager_GetRawFileSize(fp.get()));

  std::vector<char> buffer(len);

  int32_t n = OH_ResourceManager_ReadRawFile(fp.get(), buffer.data(), len);

  if (n != len) {
    std::ostringstream os;
    os << "Read file '" << filename << "' failed. Number of bytes read: " << n
       << ". Expected bytes to read: " << len;
    SHERPA_ONNX_LOGE("%s", os.str().c_str());
    return {};
  }

  return buffer;
}
#endif

std::string ResolveAbsolutePath(const std::string &path) {
  if (path.empty()) {
    return path;
  }

  try {
#ifdef _WIN32
    std::wstring wide_path = ToWideString(path);
    DWORD required_size = GetFullPathNameW(wide_path.c_str(), 0, nullptr, nullptr);
    if (required_size == 0) {
      return path;
    }
    
    std::vector<wchar_t> buffer(required_size);
    DWORD actual_size = GetFullPathNameW(
        wide_path.c_str(),
        required_size,
        buffer.data(),
        nullptr
    );
    
    if (actual_size == 0 || actual_size >= required_size) {
      return path;
    }
    
    std::wstring resolved_wide(buffer.data(), actual_size);
    return ToString(resolved_wide);
#else
    char resolved_path[PATH_MAX];
    if (realpath(path.c_str(), resolved_path) == nullptr) {
      return path;
    }
    return std::string(resolved_path);
#endif
  } catch (const std::exception&) {
    return path;
  }
}

}  // namespace sherpa_onnx
