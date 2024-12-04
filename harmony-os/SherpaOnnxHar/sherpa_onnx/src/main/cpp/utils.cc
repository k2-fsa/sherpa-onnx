// Copyright (c)  2024  Xiaomi Corporation

#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "macros.h"  // NOLINT
#include "napi.h"  // NOLINT

static std::vector<std::string> GetFilenames(NativeResourceManager *mgr,
                                             const std::string &d) {
  std::unique_ptr<RawDir, decltype(&OH_ResourceManager_CloseRawDir)> raw_dir(
      OH_ResourceManager_OpenRawDir(mgr, d.c_str()),
      &OH_ResourceManager_CloseRawDir);
  int count = OH_ResourceManager_GetRawFileCount(raw_dir.get());
  std::vector<std::string> ans;
  ans.reserve(count);
  for (int32_t i = 0; i < count; ++i) {
    std::string filename = OH_ResourceManager_GetRawFileName(raw_dir.get(), i);
    bool is_dir = OH_ResourceManager_IsRawDir(
        mgr, d.empty() ? filename.c_str() : (d + "/" + filename).c_str());
    if (is_dir) {
      auto files = GetFilenames(mgr, d.empty() ? filename : d + "/" + filename);
      for (auto &f : files) {
        ans.push_back(std::move(f));
      }
    } else {
      if (d.empty()) {
        ans.push_back(std::move(filename));
      } else {
        ans.push_back(d + "/" + filename);
      }
    }
  }

  return ans;
}

static Napi::Array ListRawFileDir(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();

  if (info.Length() != 2) {
    std::ostringstream os;
    os << "Expect only 2 arguments. Given: " << info.Length();

    Napi::TypeError::New(env, os.str()).ThrowAsJavaScriptException();

    return {};
  }

  std::unique_ptr<NativeResourceManager,
                  decltype(&OH_ResourceManager_ReleaseNativeResourceManager)>
      mgr(OH_ResourceManager_InitNativeResourceManager(env, info[0]),
          &OH_ResourceManager_ReleaseNativeResourceManager);

  if (!info[1].IsString()) {
    Napi::TypeError::New(env, "Argument 1 should be a string")
        .ThrowAsJavaScriptException();

    return {};
  }

  std::string dir = info[1].As<Napi::String>().Utf8Value();

  auto files = GetFilenames(mgr.get(), dir);
  Napi::Array ans = Napi::Array::New(env, files.size());
  for (int32_t i = 0; i != files.size(); ++i) {
    ans[i] = Napi::String::New(env, files[i]);
  }
  return ans;
}
void InitUtils(Napi::Env env, Napi::Object exports) {
  exports.Set(Napi::String::New(env, "listRawfileDir"),
              Napi::Function::New(env, ListRawFileDir));
}
