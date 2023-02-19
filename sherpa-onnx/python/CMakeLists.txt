add_subdirectory(csrc)

if(SHERPA_ONNX_ENABLE_TESTS)
  add_subdirectory(tests)
endif()
