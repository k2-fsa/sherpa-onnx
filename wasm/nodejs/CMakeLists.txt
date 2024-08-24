if(NOT $ENV{SHERPA_ONNX_IS_USING_BUILD_WASM_SH})
  message(FATAL_ERROR "Please use ./build-wasm-simd-nodejs.sh to build for wasm NodeJS")
endif()

set(exported_functions
  #tts
  PrintOfflineTtsConfig
  SherpaOnnxCreateOfflineTts
  SherpaOnnxDestroyOfflineTts
  SherpaOnnxDestroyOfflineTtsGeneratedAudio
  SherpaOnnxOfflineTtsGenerate
  SherpaOnnxOfflineTtsGenerateWithCallback
  SherpaOnnxOfflineTtsNumSpeakers
  SherpaOnnxOfflineTtsSampleRate
  SherpaOnnxWriteWave
  # streaming asr
  SherpaOnnxCreateOnlineRecognizer
  SherpaOnnxCreateOnlineStream
  SherpaOnnxDecodeOnlineStream
  SherpaOnnxDestroyOnlineRecognizer
  SherpaOnnxDestroyOnlineRecognizerResult
  SherpaOnnxDestroyOnlineStream
  SherpaOnnxDestroyOnlineStreamResultJson
  SherpaOnnxGetOnlineStreamResult
  SherpaOnnxGetOnlineStreamResultAsJson
  SherpaOnnxIsOnlineStreamReady
  SherpaOnnxOnlineStreamAcceptWaveform
  SherpaOnnxOnlineStreamInputFinished
  SherpaOnnxOnlineStreamIsEndpoint
  SherpaOnnxOnlineStreamReset
  # non-streaming ASR
  PrintOfflineRecognizerConfig
  SherpaOnnxAcceptWaveformOffline
  SherpaOnnxCreateOfflineRecognizer
  SherpaOnnxCreateOfflineStream
  SherpaOnnxDecodeMultipleOfflineStreams
  SherpaOnnxDecodeOfflineStream
  SherpaOnnxDestroyOfflineRecognizer
  SherpaOnnxDestroyOfflineRecognizerResult
  SherpaOnnxDestroyOfflineStream
  SherpaOnnxDestroyOfflineStreamResultJson
  SherpaOnnxGetOfflineStreamResult
  SherpaOnnxGetOfflineStreamResultAsJson
  # online kws
  SherpaOnnxCreateKeywordSpotter
  SherpaOnnxCreateKeywordStream
  SherpaOnnxDecodeKeywordStream
  SherpaOnnxDestroyKeywordResult
  SherpaOnnxDestroyKeywordSpotter
  SherpaOnnxGetKeywordResult
  SherpaOnnxIsKeywordStreamReady
  # VAD
  SherpaOnnxCreateCircularBuffer
  SherpaOnnxDestroyCircularBuffer
  SherpaOnnxCircularBufferPush
  SherpaOnnxCircularBufferGet
  SherpaOnnxCircularBufferFree
  SherpaOnnxCircularBufferPop
  SherpaOnnxCircularBufferSize
  SherpaOnnxCircularBufferHead
  SherpaOnnxCircularBufferReset
  SherpaOnnxCreateVoiceActivityDetector
  SherpaOnnxDestroyVoiceActivityDetector
  SherpaOnnxVoiceActivityDetectorAcceptWaveform
  SherpaOnnxVoiceActivityDetectorEmpty
  SherpaOnnxVoiceActivityDetectorDetected
  SherpaOnnxVoiceActivityDetectorPop
  SherpaOnnxVoiceActivityDetectorClear
  SherpaOnnxVoiceActivityDetectorFront
  SherpaOnnxDestroySpeechSegment
  SherpaOnnxVoiceActivityDetectorReset
  SherpaOnnxVoiceActivityDetectorFlush
  #
  SherpaOnnxFileExists
  SherpaOnnxReadWave
  SherpaOnnxFreeWave
  SherpaOnnxWriteWave
)


set(mangled_exported_functions)
foreach(x IN LISTS exported_functions)
  list(APPEND mangled_exported_functions "_${x}")
endforeach()
list(JOIN mangled_exported_functions "," all_exported_functions)

include_directories(${CMAKE_SOURCE_DIR})
set(MY_FLAGS " -s FORCE_FILESYSTEM=1 -s INITIAL_MEMORY=512MB -s ALLOW_MEMORY_GROWTH=1")
string(APPEND MY_FLAGS " -sSTACK_SIZE=10485760 ") # 10MB
string(APPEND MY_FLAGS " -sEXPORTED_FUNCTIONS=[_CopyHeap,_malloc,_free,${all_exported_functions}] ")
string(APPEND MY_FLAGS " -sEXPORTED_RUNTIME_METHODS=['ccall','stringToUTF8','setValue','getValue'] ")
string(APPEND MY_FLAGS " -sNODERAWFS=1 ")
string(APPEND MY_FLAGS " -sEXPORTED_RUNTIME_METHODS=['ccall','stringToUTF8','setValue','getValue','lengthBytesUTF8','UTF8ToString'] ")
string(APPEND MY_FLAGS " -sMODULARIZE=1 -sWASM_ASYNC_COMPILATION=0 ")

message(STATUS "MY_FLAGS: ${MY_FLAGS}")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${MY_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${MY_FLAGS}")
set(CMAKE_EXECUTBLE_LINKER_FLAGS "${CMAKE_EXECUTBLE_LINKER_FLAGS} ${MY_FLAGS}")

add_executable(sherpa-onnx-wasm-nodejs sherpa-onnx-wasm-nodejs.cc)
target_link_libraries(sherpa-onnx-wasm-nodejs sherpa-onnx-core sherpa-onnx-c-api)
install(TARGETS sherpa-onnx-wasm-nodejs DESTINATION bin/wasm/nodejs)

install(
  FILES
  ${CMAKE_SOURCE_DIR}/wasm/asr/sherpa-onnx-asr.js
  ${CMAKE_SOURCE_DIR}/wasm/tts/sherpa-onnx-tts.js
  ${CMAKE_SOURCE_DIR}/wasm/kws/sherpa-onnx-kws.js
  ${CMAKE_SOURCE_DIR}/wasm/vad/sherpa-onnx-vad.js
  ${CMAKE_SOURCE_DIR}/wasm/nodejs/sherpa-onnx-wave.js
    "$<TARGET_FILE_DIR:sherpa-onnx-wasm-nodejs>/sherpa-onnx-wasm-nodejs.js"
    "$<TARGET_FILE_DIR:sherpa-onnx-wasm-nodejs>/sherpa-onnx-wasm-nodejs.wasm"
  DESTINATION
    bin/wasm/nodejs
)
