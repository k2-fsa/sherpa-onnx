// sherpa-onnx/csrc/session-test.cc
//
// Copyright (c)  2026  Kevin Castillo

#include "sherpa-onnx/csrc/session.h"

#include <cstdio>
#include <fstream>
#include <string>

#include "gtest/gtest.h"

namespace sherpa_onnx {

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
  const std::string config_path = "session-test-provider-config.txt";
  {
    std::ofstream os(config_path);
    os << "# comment line\n";
    os << "SessionConfig.mlas.disable_kleidiai=1\n";
    os << "SessionConfig.session.disable_prepacking = 1\n";
    os << "EnableCpuMemArena=0\n";
    os << "SPACEMIT_EP_USE_GLOBAL_INTRA_THREAD=1\n";
    os << "SessionConfig.=1\n";
  }

  auto sess_opts = GetSessionOptionsImpl(1, "cpu:" + config_path);

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

  std::remove(config_path.c_str());
}

}  // namespace sherpa_onnx
