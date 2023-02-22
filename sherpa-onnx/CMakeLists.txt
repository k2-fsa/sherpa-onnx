add_subdirectory(csrc)
if(SHERPA_ONNX_ENABLE_PYTHON)
  add_subdirectory(python)
endif()

if(SHERPA_ONNX_ENABLE_JNI)
  add_subdirectory(jni)
endif()
