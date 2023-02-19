add_subdirectory(csrc)
if(SHERPA_ONNX_ENABLE_PYTHON)
  add_subdirectory(python)
endif()
