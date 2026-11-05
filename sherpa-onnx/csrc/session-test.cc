// sherpa-onnx/csrc/session-test.cc
//
// Copyright (c)  2026  Kevin Castillo

#include "sherpa-onnx/csrc/session.h"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <string>
#include <unordered_map>

#if defined(_WIN32)
#include <windows.h>
#else
#include <unistd.h>
#endif

#include "gtest/gtest.h"
#include "sherpa-onnx/csrc/provider.h"

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

TEST(CudaProviderOptions, DefaultsWhenConfigIsEmpty) {
  auto opts = BuildCudaProviderOptions({}, 2, OrtCudnnConvAlgoSearchHeuristic);
  EXPECT_EQ(opts.at("device_id"), "2");
  EXPECT_EQ(opts.at("cudnn_conv_algo_search"), "HEURISTIC");
  EXPECT_EQ(opts.size(), 2);
}

TEST(CudaProviderOptions, ConfigKeysWinAndPassThrough) {
  std::unordered_map<std::string, std::string> config = {
      {"device_id", "5"},
      {"gpu_mem_limit", "2147483648"},
      {"arena_extend_strategy", "kSameAsRequested"},
      {"DEBUG", "1"},
  };
  auto opts =
      BuildCudaProviderOptions(config, 0, OrtCudnnConvAlgoSearchExhaustive);
  EXPECT_EQ(opts.at("device_id"), "5");
  EXPECT_EQ(opts.at("cudnn_conv_algo_search"), "EXHAUSTIVE");
  EXPECT_EQ(opts.at("gpu_mem_limit"), "2147483648");
  EXPECT_EQ(opts.at("arena_extend_strategy"), "kSameAsRequested");
  EXPECT_EQ(opts.count("DEBUG"), 0);
}

TEST(CudaProviderOptions, MapsEveryAlgoSearchValue) {
  EXPECT_EQ(BuildCudaProviderOptions({}, 0, OrtCudnnConvAlgoSearchExhaustive)
                .at("cudnn_conv_algo_search"),
            "EXHAUSTIVE");
  EXPECT_EQ(BuildCudaProviderOptions({}, 0, OrtCudnnConvAlgoSearchHeuristic)
                .at("cudnn_conv_algo_search"),
            "HEURISTIC");
  EXPECT_EQ(BuildCudaProviderOptions({}, 0, OrtCudnnConvAlgoSearchDefault)
                .at("cudnn_conv_algo_search"),
            "DEFAULT");
}

TEST(CudaDeviceId, UsesMergedDefaultsAndOverrides) {
  int32_t device_id = -1;
  ASSERT_TRUE(GetCudaDeviceId(
      BuildCudaProviderOptions({}, 0, OrtCudnnConvAlgoSearchHeuristic),
      &device_id));
  EXPECT_EQ(device_id, 0);
  ASSERT_TRUE(GetCudaDeviceId(
      BuildCudaProviderOptions({}, 2, OrtCudnnConvAlgoSearchHeuristic),
      &device_id));
  EXPECT_EQ(device_id, 2);
  ASSERT_TRUE(GetCudaDeviceId(
      BuildCudaProviderOptions({{"device_id", "1"}}, 2,
                               OrtCudnnConvAlgoSearchHeuristic),
      &device_id));
  EXPECT_EQ(device_id, 1);
}

TEST(CudaDeviceId, AcceptsNonNegativeInt32) {
  for (const auto &value : {"0", "1", "2147483647"}) {
    SCOPED_TRACE(value);
    int32_t device_id = -1;
    ASSERT_TRUE(GetCudaDeviceId({{"device_id", value}}, &device_id));
    EXPECT_EQ(std::to_string(device_id), value);
  }
}

TEST(CudaDeviceId, RejectsInvalidIdsWithoutChangingOutput) {
  int32_t device_id = 7;
  EXPECT_FALSE(GetCudaDeviceId({}, &device_id));
  EXPECT_EQ(device_id, 7);
  for (const auto &value : {"", "-1", "2147483648", "1x", "1.0"}) {
    SCOPED_TRACE(value);
    EXPECT_FALSE(GetCudaDeviceId({{"device_id", value}}, &device_id));
    EXPECT_EQ(device_id, 7);
  }
}

TEST(CudaDeviceId, RejectsEmbeddedNulWithoutTruncating) {
  int32_t device_id = 7;
  EXPECT_FALSE(GetCudaDeviceId({{"device_id", std::string("1\0x", 3)}},
                               &device_id));
  EXPECT_EQ(device_id, 7);
}

TEST(CudaProviderInfo, NonCudaProvidersResetPreviousState) {
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "cuda-session-test");
  TempConfigFile config("device_id=1\n");
  ASSERT_TRUE(config.Ok());
  for (const std::string &provider :
       {std::string("cpu"), std::string(""), std::string("unknown-provider"),
        "cpu:" + config.Path()}) {
    SCOPED_TRACE(provider);
    CudaProviderInfo info{true, 7};
    auto opts = GetSessionOptionsImpl(1, provider, nullptr, &info);
    EXPECT_FALSE(info.enabled);
    EXPECT_EQ(info.device_id, 0);
  }
}

TEST(CudaProviderInfo, UnavailableCudaNeverEnablesBinding) {
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "cuda-session-test");
  const auto providers = Ort::GetAvailableProviders();
  if (std::find(providers.begin(), providers.end(), "CUDAExecutionProvider") !=
      providers.end()) {
    GTEST_SKIP() << "Requires ONNX Runtime without CUDA EP";
  }
  TempConfigFile empty_config("");
  TempConfigFile config("device_id=1\n");
  ASSERT_TRUE(empty_config.Ok());
  ASSERT_TRUE(config.Ok());
  for (const std::string &provider :
       {std::string("cuda"), "cuda:" + empty_config.Path(),
        "  CuDa : " + config.Path() + "  "}) {
    SCOPED_TRACE(provider);
    CudaProviderInfo info{true, 7};
    auto opts = GetSessionOptionsImpl(1, provider, nullptr, &info);
    EXPECT_FALSE(info.enabled);
    EXPECT_EQ(info.device_id, 0);
  }
}

TEST(CudaProviderInfo, RejectedCudaOptionNeverEnablesBinding) {
#if ORT_API_VERSION >= 12
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "cuda-session-test");
  const auto providers = Ort::GetAvailableProviders();
  if (std::find(providers.begin(), providers.end(), "CUDAExecutionProvider") ==
      providers.end()) {
    GTEST_SKIP() << "Requires CUDA EP to exercise option rejection";
  }
  TempConfigFile config("sherpa_test_invalid_cuda_option=1\n");
  ASSERT_TRUE(config.Ok());
  CudaProviderInfo info{true, 7};
  auto opts = GetSessionOptionsImpl(1, "cuda:" + config.Path(), nullptr, &info);
  EXPECT_FALSE(info.enabled);
  EXPECT_EQ(info.device_id, 0);
#else
  GTEST_SKIP() << "Requires CUDA V2 provider options";
#endif
}

TEST(Provider, CudaConfigFileDoesNotThrowWithoutCudaEp) {
  // Without a CUDA-enabled onnxruntime this falls back to the CPU provider;
  // it must still parse the config file without crashing.
  TempConfigFile config(
      "gpu_mem_limit=2147483648\n"
      "arena_extend_strategy=kSameAsRequested\n");
  ASSERT_TRUE(config.Ok());
  EXPECT_NO_THROW({
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "cuda-session-test");
    auto sess_opts = GetSessionOptionsImpl(1, "cuda:" + config.Path());
    (void)sess_opts;
  });
}

TEST(Provider, ConvertsOpenVINOCaseInsensitively) {
  EXPECT_EQ(StringToProvider("openvino"), Provider::kOpenVINO);
  EXPECT_EQ(StringToProvider("OpenVINO"), Provider::kOpenVINO);
}

TEST(Provider, OpenVINOSessionOptionsDoNotThrow) {
  EXPECT_NO_THROW({
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "openvino-session-test");
    auto sess_opts = GetSessionOptionsImpl(1, "openvino");
    (void)sess_opts;
  });
}

}  // namespace sherpa_onnx
