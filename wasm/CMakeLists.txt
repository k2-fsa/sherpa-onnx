if(SHERPA_ONNX_ENABLE_WASM_TTS)
  add_subdirectory(tts)
endif()

if(SHERPA_ONNX_ENABLE_WASM_ASR)
  add_subdirectory(asr)
endif()

if(SHERPA_ONNX_ENABLE_WASM_NODEJS)
  add_subdirectory(nodejs)
endif()
