// sherpa-onnx/csrc/fst-utils.cc
//
// Copyright (c)  2024  Xiaomi Corporation

#include "sherpa-onnx/csrc/fst-utils.h"

#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

#include "fst/extensions/far/far.h"
#include "kaldifst/csrc/kaldi-fst-io.h"
#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {

namespace {

std::string WriteFarBufferToTempFile(const std::vector<char> &buffer) {
#ifdef _WIN32
  char temp_path[MAX_PATH];
  DWORD path_size = GetTempPathA(MAX_PATH, temp_path);
  if (path_size == 0 || path_size > MAX_PATH) {
    SHERPA_ONNX_LOGE("Failed to get a temporary directory for FAR extraction");
    SHERPA_ONNX_EXIT(-1);
  }

  char temp_filename[MAX_PATH];
  if (GetTempFileNameA(temp_path, "sof", 0, temp_filename) == 0) {
    SHERPA_ONNX_LOGE("Failed to create a temporary FAR file");
    SHERPA_ONNX_EXIT(-1);
  }

  std::string filename = temp_filename;
#else
  char temp_filename[] = "/tmp/sherpa-onnx-far-XXXXXX";
  int fd = mkstemp(temp_filename);
  if (fd == -1) {
    SHERPA_ONNX_LOGE("Failed to create a temporary FAR file");
    SHERPA_ONNX_EXIT(-1);
  }
  close(fd);

  std::string filename = temp_filename;
#endif

  std::ofstream os(filename, std::ios::binary);
  if (!os.is_open()) {
    SHERPA_ONNX_LOGE("Failed to open temporary FAR file '%s' for writing",
                     filename.c_str());
    SHERPA_ONNX_EXIT(-1);
  }

  os.write(buffer.data(), buffer.size());
  if (!os.good()) {
    SHERPA_ONNX_LOGE("Failed to write temporary FAR file '%s'",
                     filename.c_str());
    SHERPA_ONNX_EXIT(-1);
  }

  return filename;
}

struct TempFileGuard {
  explicit TempFileGuard(std::string filename) : filename(std::move(filename)) {}

  ~TempFileGuard() {
    if (!filename.empty()) {
      std::remove(filename.c_str());
    }
  }

  std::string filename;
};

}  // namespace

// This function is copied from kaldi.
//
// @param filename Path to a StdVectorFst or StdConstFst graph
// @return The caller should free the returned pointer using `delete` to
//         avoid memory leak.
fst::Fst<fst::StdArc> *ReadGraph(const std::string &filename) {
  // read decoding network FST
  std::ifstream is(filename, std::ios::binary);
  if (!is.good()) {
    SHERPA_ONNX_LOGE("Could not open decoding-graph FST %s", filename.c_str());
  }

  fst::FstHeader hdr;
  if (!hdr.Read(is, "<unknown>")) {
    SHERPA_ONNX_LOGE("Reading FST: error reading FST header.");
  }

  if (hdr.ArcType() != fst::StdArc::Type()) {
    SHERPA_ONNX_LOGE("FST with arc type %s not supported",
                     hdr.ArcType().c_str());
  }
  fst::FstReadOptions ropts("<unspecified>", &hdr);

  fst::Fst<fst::StdArc> *decode_fst = nullptr;

  if (hdr.FstType() == "vector") {
    decode_fst = fst::VectorFst<fst::StdArc>::Read(is, ropts);
  } else if (hdr.FstType() == "const") {
    decode_fst = fst::ConstFst<fst::StdArc>::Read(is, ropts);
  } else {
    SHERPA_ONNX_LOGE("Reading FST: unsupported FST type: %s",
                     hdr.FstType().c_str());
  }

  if (decode_fst == nullptr) {  // fst code will warn.
    SHERPA_ONNX_LOGE("Error reading FST (after reading header).");
    return nullptr;
  } else {
    return decode_fst;
  }
}

std::vector<std::unique_ptr<fst::StdConstFst>> ReadFstsFromFar(
    const std::vector<char> &buffer) {
  std::vector<std::unique_ptr<fst::StdConstFst>> ans;
  auto filename = WriteFarBufferToTempFile(buffer);
  TempFileGuard guard(filename);

  std::unique_ptr<fst::FarReader<fst::StdArc>> reader(
      fst::FarReader<fst::StdArc>::Open(filename));
  if (!reader) {
    SHERPA_ONNX_LOGE("Failed to open FAR data");
    SHERPA_ONNX_EXIT(-1);
  }

  for (; !reader->Done(); reader->Next()) {
    ans.emplace_back(fst::CastOrConvertToConstFst(reader->GetFst()->Copy()));
  }

  return ans;
}

}  // namespace sherpa_onnx
