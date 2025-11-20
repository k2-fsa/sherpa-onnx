// sherpa-onnx/jni/common.cc
//
// Copyright (c)  2025  Xiaomi Corporation
#include "sherpa-onnx/jni/common.h"

#include <stdlib.h>

#include <string>

#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {

/* For qnn to load libQnnHtpVxxSkel.so, e.g., libQnnHtpV81Skel.so file

https://workbench.aihub.qualcomm.com/docs/hub/faq.html#why-am-i-seeing-error-1008-when-trying-to-use-htp
 */
void PrependAdspLibraryPath(const std::string &new_path) {
  const char *old_path = getenv("ADSP_LIBRARY_PATH");
  std::string updated_path;

  if (old_path && !std::string(old_path).empty()) {
    // Caution(fangjun):
    // 1. Must use ; here, not :
    // 2. Must use prepend, not append
    updated_path = new_path + ";" + std::string(old_path);
  } else {
    updated_path = new_path;  // no old path
  }

  if (setenv("ADSP_LIBRARY_PATH", updated_path.c_str(), 1) != 0) {
    SHERPA_ONNX_LOGE("Failed to set ADSP_LIBRARY_PATH to '%s'",
                     updated_path.c_str());
  } else {
    SHERPA_ONNX_LOGE("Successfully set ADSP_LIBRARY_PATH to '%s'",
                     updated_path.c_str());
  }
  /*
You will see something like the following:

Successfully set ADSP_LIBRARY_PATH to
'/data/app/~~pHS2-9SwVjl9ma3cIKtj-g==/com.k2fsa.sherpa.onnx.simulate.streaming.asr-ejCDb8LodsnyK5cr3SvGjA==/lib/arm64;/odm/lib/rfsa/adsp;/vendor/lib/rfsa/adsp/;/system/lib/rfsa/adsp;/system/vendor/lib/rfsa/adsp;/dsp'

   */
}

}  // namespace sherpa_onnx
