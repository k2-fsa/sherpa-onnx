if(NOT $ENV{SHERPA_ONNX_IS_USING_BUILD_WASM_SH})
  message(FATAL_ERROR "Please use ./build-wasm-simd-asr.sh to build for wasm ASR")
endif()

if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/assets/encoder.onnx")
  message(FATAL_ERROR "Please read ${CMAKE_CURRENT_SOURCE_DIR}/assets/README.md before you continue")
endif()

set(exported_functions
  MyPrint
  # online ASR
  AcceptWaveform
  CreateOnlineRecognizer
  CreateOnlineStream
  DecodeOnlineStream
  DestroyOfflineStreamResultJson
  DestroyOnlineRecognizer
  DestroyOnlineRecognizerResult
  DestroyOnlineStream
  DestroyOnlineStreamResultJson
  GetOfflineStreamResultAsJson
  GetOnlineStreamResult
  GetOnlineStreamResultAsJson
  InputFinished
  IsEndpoint
  IsOnlineStreamReady
  Reset
  #
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
string(APPEND MY_FLAGS "--preload-file ${CMAKE_CURRENT_SOURCE_DIR}/assets@. ")
string(APPEND MY_FLAGS " -sEXPORTED_RUNTIME_METHODS=['ccall','stringToUTF8','setValue','getValue','lengthBytesUTF8','UTF8ToString'] ")

message(STATUS "MY_FLAGS: ${MY_FLAGS}")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${MY_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${MY_FLAGS}")
set(CMAKE_EXECUTBLE_LINKER_FLAGS "${CMAKE_EXECUTBLE_LINKER_FLAGS} ${MY_FLAGS}")

if (NOT CMAKE_EXECUTABLE_SUFFIX STREQUAL ".js")
  message(FATAL_ERROR "The default suffix for building executables should be .js!")
endif()
# set(CMAKE_EXECUTABLE_SUFFIX ".html")

add_executable(sherpa-onnx-wasm-main-asr sherpa-onnx-wasm-main-asr.cc)
target_link_libraries(sherpa-onnx-wasm-main-asr sherpa-onnx-c-api)
install(TARGETS sherpa-onnx-wasm-main-asr DESTINATION bin/wasm/asr)

install(
  FILES
    "$<TARGET_FILE_DIR:sherpa-onnx-wasm-main-asr>/sherpa-onnx-wasm-main-asr.js"
    "index.html"
    "sherpa-onnx-asr.js"
    "app-asr.js"
    "$<TARGET_FILE_DIR:sherpa-onnx-wasm-main-asr>/sherpa-onnx-wasm-main-asr.wasm"
    "$<TARGET_FILE_DIR:sherpa-onnx-wasm-main-asr>/sherpa-onnx-wasm-main-asr.data"
  DESTINATION
    bin/wasm/asr
)
