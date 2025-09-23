// sherpa-onnx/csrc/offline-tts-zipvoice-frontend-test.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-tts-zipvoice-frontend.h"

#include "espeak-ng/speak_lib.h"
#include "gtest/gtest.h"
#include "phoneme_ids.hpp"
#include "phonemize.hpp"
#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {

TEST(ZipVoiceFrontend, Case1) {
  std::string data_dir = "../zipvoice/espeak-ng-data";
  if (!FileExists(data_dir + "/en_dict")) {
    SHERPA_ONNX_LOGE("%s/en_dict does not exist. Skipping test",
                     data_dir.c_str());
    return;
  }

  if (!FileExists(data_dir + "/phontab")) {
    SHERPA_ONNX_LOGE("%s/phontab does not exist. Skipping test",
                     data_dir.c_str());
    return;
  }

  if (!FileExists(data_dir + "/phonindex")) {
    SHERPA_ONNX_LOGE("%s/phonindex does not exist. Skipping test",
                     data_dir.c_str());
    return;
  }

  if (!FileExists(data_dir + "/phondata")) {
    SHERPA_ONNX_LOGE("%s/phondata does not exist. Skipping test",
                     data_dir.c_str());
    return;
  }

  if (!FileExists(data_dir + "/intonations")) {
    SHERPA_ONNX_LOGE("%s/intonations does not exist. Skipping test",
                     data_dir.c_str());
    return;
  }

  std::string pinyin_dict = data_dir + "/../pinyin.dict";
  if (!FileExists(pinyin_dict)) {
    SHERPA_ONNX_LOGE("%s does not exist. Skipping test", pinyin_dict.c_str());
    return;
  }

  std::string tokens_file = data_dir + "/../tokens.txt";
  if (!FileExists(tokens_file)) {
    SHERPA_ONNX_LOGE("%s does not exist. Skipping test", tokens_file.c_str());
    return;
  }

  auto frontend = OfflineTtsZipvoiceFrontend(
      tokens_file, data_dir, pinyin_dict,
      OfflineTtsZipvoiceModelMetaData{.use_espeak = true, .use_pinyin = true},
      true);

  std::string text = "how are you doing?";
  std::vector<sherpa_onnx::TokenIDs> ans =
      frontend.ConvertTextToTokenIds(text, "en-us");

  text = "这是第一句。这是第二句。";
  ans = frontend.ConvertTextToTokenIds(text, "en-us");

  text =
      "这是第一句。这是第二句。<pin1><yin2>测试 [S1]and hello "
      "world[S2]这是第三句。";
  ans = frontend.ConvertTextToTokenIds(text, "en-us");
}

}  // namespace sherpa_onnx
