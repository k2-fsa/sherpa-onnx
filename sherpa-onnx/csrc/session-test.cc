// sherpa-onnx/csrc/session-test.cc
//
// Copyright (c)  2026  Kevin Castillo

#include "sherpa-onnx/csrc/session.h"

#include <cstdio>
#include <fstream>
#include <string>

#if defined(_WIN32)
#include <windows.h>
#else
#include <unistd.h>
#endif

#include "gtest/gtest.h"

namespace sherpa_onnx {

namespace {

// RAII temporary provider-config file, following the TempFile pattern in
// wave-reader-test.cc (mkstemp / GetTempFileNameA).
class TempConfigFile {
 public:
  explicit TempConfigFile(const std::string &contents) {
#if defined(_WIN32)
    char temp_path[MAX_PATH];
    char temp_file[MAX_PATH];
    DWORD path_len = GetTempPathA(MAX_PATH, temp_path);
    if (path_len > 0 && path_len < MAX_PATH &&
        GetTempFileNameA(temp_path, "soc", 0, temp_file) != 0) {
      path_ = temp_file;
    }
#else
    char temp_template[] = "/tmp/sherpa_onnx_session_test_XXXXXX";
    int fd = mkstemp(temp_template);
    if (fd != -1) {
      close(fd);
      path_ = temp_template;
    }
#endif
    if (path_.empty()) {
      return;
    }
    std::ofstream os(path_);
    ok_ = os.is_open();
    os << contents;
  }

  ~TempConfigFile() {
    if (!path_.empty()) {
      std::remove(path_.c_str());
    }
  }

  const std::string &Path() const { return path_; }
  bool Ok() const { return ok_; }

 private:
  std::string path_;
  bool ok_ = false;
};

}  // namespace

static bool HasConfigEntry(const Ort::SessionOptions &sess_opts,
                           const char *key) {
  int has = 0;
  Ort::ThrowOnError(Ort::GetApi().HasSessionConfigEntry(
      static_cast<const OrtSessionOptions *>(sess_opts), key, &has));
  return has != 0;
}

static std::string GetConfigEntry(const Ort::SessionOptions &sess_opts,
                                  const char *key) {
  size_t size = 0;
  Ort::ThrowOnError(Ort::GetApi().GetSessionConfigEntry(
      static_cast<const OrtSessionOptions *>(sess_opts), key, nullptr, &size));
  std::string value(size, '\0');
  Ort::ThrowOnError(Ort::GetApi().GetSessionConfigEntry(
      static_cast<const OrtSessionOptions *>(sess_opts), key, value.data(),
      &size));
  if (!value.empty() && value.back() == '\0') {
    value.pop_back();
  }
  return value;
}

TEST(SessionConfigPassthrough, ForwardsPrefixedKeysAndIgnoresOthers) {
  TempConfigFile config(
      "# comment line\n"
      "SessionConfig.mlas.disable_kleidiai=1\n"
      "SessionConfig.session.disable_prepacking = 1\n"
      "EnableCpuMemArena=0\n"
      "SPACEMIT_EP_USE_GLOBAL_INTRA_THREAD=1\n"
      "SessionConfig.=1\n");
  ASSERT_TRUE(config.Ok());

  auto sess_opts = GetSessionOptionsImpl(1, "cpu:" + config.Path());

  // SessionConfig.-prefixed keys are forwarded with the prefix stripped.
  EXPECT_TRUE(HasConfigEntry(sess_opts, "mlas.disable_kleidiai"));
  EXPECT_EQ(GetConfigEntry(sess_opts, "mlas.disable_kleidiai"), "1");
  EXPECT_TRUE(HasConfigEntry(sess_opts, "session.disable_prepacking"));
  EXPECT_EQ(GetConfigEntry(sess_opts, "session.disable_prepacking"), "1");

  // The prefixed spelling itself is not registered.
  EXPECT_FALSE(
      HasConfigEntry(sess_opts, "SessionConfig.mlas.disable_kleidiai"));

  // Known sherpa-onnx keys are consumed, not forwarded.
  EXPECT_FALSE(HasConfigEntry(sess_opts, "EnableCpuMemArena"));

  // Un-prefixed provider-specific keys are left alone.
  EXPECT_FALSE(
      HasConfigEntry(sess_opts, "SPACEMIT_EP_USE_GLOBAL_INTRA_THREAD"));

  // "SessionConfig.=1" (empty key after the prefix) is ignored: nothing is
  // registered under the empty key or under the raw spelling.
  EXPECT_FALSE(HasConfigEntry(sess_opts, ""));
  EXPECT_FALSE(HasConfigEntry(sess_opts, "SessionConfig."));
}

}  // namespace sherpa_onnx
