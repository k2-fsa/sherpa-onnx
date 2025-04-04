if(NOT $ENV{SHERPA_ONNX_IS_USING_BUILD_WASM_SH})
  message(FATAL_ERROR "Please use ./build-wasm-combined.sh to build")
endif()

# Collect all exported functions from all modules
set(exported_functions
  # Core utilities
  CopyHeap
  MyPrintOnlineASR
  MyPrintVAD
  MyPrintTTS
  MyPrintSpeakerDiarization
  MyPrintSpeechEnhancement
  MyPrintKeywordSpotting
  SherpaOnnxFileExists

  # Online ASR
  SherpaOnnxCreateOnlineRecognizer
  SherpaOnnxCreateOnlineStream
  SherpaOnnxDecodeOnlineStream
  SherpaOnnxDestroyOfflineStreamResultJson
  SherpaOnnxDestroyOnlineRecognizer
  SherpaOnnxDestroyOnlineRecognizerResult
  SherpaOnnxDestroyOnlineStream
  SherpaOnnxDestroyOnlineStreamResultJson
  SherpaOnnxGetOfflineStreamResultAsJson
  SherpaOnnxGetOnlineStreamResult
  SherpaOnnxGetOnlineStreamResultAsJson
  SherpaOnnxIsOnlineStreamReady
  SherpaOnnxOnlineStreamAcceptWaveform
  SherpaOnnxOnlineStreamInputFinished
  SherpaOnnxOnlineStreamIsEndpoint
  SherpaOnnxOnlineStreamReset
  
  # Offline ASR
  SherpaOnnxCreateOfflineRecognizer
  SherpaOnnxCreateOfflineStream
  SherpaOnnxDecodeOfflineStream
  SherpaOnnxDecodeMultipleOfflineStreams
  SherpaOnnxDestroyOfflineRecognizer
  SherpaOnnxDestroyOfflineRecognizerResult
  SherpaOnnxDestroyOfflineStream
  SherpaOnnxAcceptWaveformOffline
  SherpaOnnxGetOfflineStreamResult
  
  # TTS
  SherpaOnnxCreateOfflineTts
  SherpaOnnxDestroyOfflineTts
  SherpaOnnxDestroyOfflineTtsGeneratedAudio
  SherpaOnnxOfflineTtsGenerate
  SherpaOnnxOfflineTtsGenerateWithCallback
  SherpaOnnxOfflineTtsSampleRate
  SherpaOnnxOfflineTtsNumSpeakers
  SherpaOnnxWriteWave
  
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
  
  # KWS
  SherpaOnnxCreateKeywordSpotter
  SherpaOnnxDestroyKeywordSpotter
  SherpaOnnxCreateKeywordStream
  SherpaOnnxIsKeywordStreamReady
  SherpaOnnxDecodeKeywordStream
  SherpaOnnxResetKeywordStream
  SherpaOnnxGetKeywordResult
  SherpaOnnxDestroyKeywordResult
)

set(mangled_exported_functions)
foreach(x IN LISTS exported_functions)
  list(APPEND mangled_exported_functions "_${x}")
endforeach()
list(JOIN mangled_exported_functions "," all_exported_functions)

include_directories(${CMAKE_SOURCE_DIR})
set(MY_FLAGS " -s FORCE_FILESYSTEM=1 -s INITIAL_MEMORY=512MB -s ALLOW_MEMORY_GROWTH=1")
string(APPEND MY_FLAGS " -sSTACK_SIZE=10485760 ") # 10MB
string(APPEND MY_FLAGS " -sASYNCIFY=1 -sFETCH=1 ") # For async loading
string(APPEND MY_FLAGS " -sEXPORTED_FUNCTIONS=[_malloc,_free,${all_exported_functions}] ")
# No preloaded assets - all models will be loaded dynamically
string(APPEND MY_FLAGS " -sEXPORTED_RUNTIME_METHODS=['ccall','stringToUTF8','setValue','getValue','lengthBytesUTF8','UTF8ToString','FS'] ")

message(STATUS "MY_FLAGS: ${MY_FLAGS}")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${MY_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${MY_FLAGS}")
set(CMAKE_EXECUTBLE_LINKER_FLAGS "${CMAKE_EXECUTBLE_LINKER_FLAGS} ${MY_FLAGS}")

add_executable(sherpa-onnx-wasm-combined sherpa-onnx-wasm-combined.cc)
target_link_libraries(sherpa-onnx-wasm-combined sherpa-onnx-c-api)
install(TARGETS sherpa-onnx-wasm-combined DESTINATION bin/wasm/combined)

install(
  FILES
    "$<TARGET_FILE_DIR:sherpa-onnx-wasm-combined>/sherpa-onnx-wasm-combined.js"
    "index.html"
    "sherpa-onnx-combined.js"
    "$<TARGET_FILE_DIR:sherpa-onnx-wasm-combined>/sherpa-onnx-wasm-combined.wasm"
  DESTINATION
    bin/wasm/combined
)

# Add option to install to original repo
option(SHERPA_ONNX_INSTALL_TO_REPO "Install compiled WASM files to original repo directory" OFF)
set(SHERPA_ONNX_REPO_PATH "${CMAKE_SOURCE_DIR}/wasm/combined" CACHE PATH "Path to original repo wasm directory")

if(SHERPA_ONNX_INSTALL_TO_REPO)
  # Add a custom target that will run after the installation
  add_custom_target(install_to_repo ALL
    COMMAND ${CMAKE_COMMAND} -E echo "Installing to original repo at ${SHERPA_ONNX_REPO_PATH}..."
    COMMAND ${CMAKE_COMMAND} -E make_directory ${SHERPA_ONNX_REPO_PATH}
    
    # Copy the JS file
    COMMAND ${CMAKE_COMMAND} 
      -DSRC_DIR=${CMAKE_BINARY_DIR}/bin
      -DDEST_DIR=${SHERPA_ONNX_REPO_PATH}
      -DCOPY_FILES="sherpa-onnx-wasm-combined.js"
      -P ${CMAKE_CURRENT_SOURCE_DIR}/copy_with_confirm.cmake
    
    # Copy the WASM file
    COMMAND ${CMAKE_COMMAND} 
      -DSRC_DIR=${CMAKE_BINARY_DIR}/bin
      -DDEST_DIR=${SHERPA_ONNX_REPO_PATH}
      -DCOPY_FILES="sherpa-onnx-wasm-combined.wasm"
      -P ${CMAKE_CURRENT_SOURCE_DIR}/copy_with_confirm.cmake
    
    # Copy the index.html file
    COMMAND ${CMAKE_COMMAND}
      -DSRC_DIR=${CMAKE_CURRENT_SOURCE_DIR}
      -DDEST_DIR=${SHERPA_ONNX_REPO_PATH}
      -DCOPY_FILES="index.html"
      -P ${CMAKE_CURRENT_SOURCE_DIR}/copy_with_confirm.cmake
    
    # Copy the JS library file
    COMMAND ${CMAKE_COMMAND}
      -DSRC_DIR=${CMAKE_CURRENT_SOURCE_DIR}
      -DDEST_DIR=${SHERPA_ONNX_REPO_PATH}
      -DCOPY_FILES="sherpa-onnx-combined.js"
      -P ${CMAKE_CURRENT_SOURCE_DIR}/copy_with_confirm.cmake
    
    DEPENDS sherpa-onnx-wasm-combined
    COMMENT "Checking and installing WASM files to original repo"
  )
endif() 