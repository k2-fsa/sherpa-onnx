if(NOT $ENV{SHERPA_ONNX_IS_USING_BUILD_WASM_SH})
  message(FATAL_ERROR "Please use ./build-wasm-simd-tts.sh to build for wasm TTS")
endif()

if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/assets/tokens.txt" AND
   NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/assets/lm_flow.int8.onnx" AND
   NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/assets/unicode_indexer.bin")
  message(FATAL_ERROR "Please read ${CMAKE_CURRENT_SOURCE_DIR}/assets/README.md before you continue")
endif()

# Text frontend: check which option of assets/README.md is set up. Neither the
# model package's espeak-ng-data nor anything in sherpa-onnx provides one, so
# without these files the demo cannot convert text at all.
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/piper-phonemize/piper-phonemize.js")
  message(STATUS "Text frontend: piper-phonemize (option 1)")
elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/assets/lexicon-en-us.txt")
  message(STATUS "Text frontend: lexicon (option 2: assets/lexicon-en-us.txt)")
elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/assets/lexicon.txt")
  message(STATUS "Text frontend: lexicon (option 2: assets/lexicon.txt)")
elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/assets/lm_flow.int8.onnx")
  message(STATUS "Text frontend: built into the pocket model")
elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/assets/unicode_indexer.bin")
  message(STATUS "Text frontend: built into the supertonic model")
else()
  message(FATAL_ERROR
    "No text frontend found: please read ${CMAKE_CURRENT_SOURCE_DIR}/assets/README.md "
    "before you continue. Either the piper-phonemize wasm bundle (option 1) or a "
    "lexicon file (option 2) is required; the pocket and supertonic models bring "
    "their own.")
endif()

set(exported_functions
  MyPrint
  SherpaOnnxCreateOfflineTts
  SherpaOnnxDestroyOfflineTts
  SherpaOnnxDestroyOfflineTtsGeneratedAudio
  SherpaOnnxDestroyOfflineTtsLang
  SherpaOnnxOfflineTtsGenerate
  SherpaOnnxOfflineTtsGenerateWithCallback
  SherpaOnnxOfflineTtsGenerateWithConfig
  SherpaOnnxOfflineTtsLang
  SherpaOnnxOfflineTtsNumSpeakers
  SherpaOnnxOfflineTtsSampleRate
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
string(APPEND MY_FLAGS " -sALLOW_TABLE_GROWTH ")
string(APPEND MY_FLAGS " -sEXPORTED_FUNCTIONS=[_CopyHeap,_malloc,_free,${all_exported_functions}] ")
string(APPEND MY_FLAGS "--preload-file ${CMAKE_CURRENT_SOURCE_DIR}/assets@. ")
string(APPEND MY_FLAGS " -sEXPORTED_RUNTIME_METHODS=['ccall','stringToUTF8','setValue','getValue','lengthBytesUTF8','UTF8ToString','HEAPU8','HEAP16','HEAP32','HEAPU32','HEAPF32','HEAPF64','addFunction','removeFunction'] ")

message(STATUS "MY_FLAGS: ${MY_FLAGS}")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${MY_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${MY_FLAGS}")
set(CMAKE_EXECUTABLE_LINKER_FLAGS "${CMAKE_EXECUTABLE_LINKER_FLAGS} ${MY_FLAGS}")

if (NOT CMAKE_EXECUTABLE_SUFFIX STREQUAL ".js")
  message(FATAL_ERROR "The default suffix for building executables should be .js!")
endif()
# set(CMAKE_EXECUTABLE_SUFFIX ".html")

add_executable(sherpa-onnx-wasm-main-tts sherpa-onnx-wasm-main-tts.cc)
target_link_libraries(sherpa-onnx-wasm-main-tts sherpa-onnx-c-api)
install(TARGETS sherpa-onnx-wasm-main-tts DESTINATION bin/wasm/tts)
# The piper-phonemize wasm bundle (option 1: the default text frontend);
# see assets/README.md. Absent for the lexicon-only models.
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/piper-phonemize DESTINATION bin/wasm/tts OPTIONAL)

install(
  FILES
    "$<TARGET_FILE_DIR:sherpa-onnx-wasm-main-tts>/sherpa-onnx-wasm-main-tts.js"
    "index.html"
    "sherpa-onnx-tts.js"
    "sherpa-onnx-tts.worker.js"
    "app-tts.js"
    "$<TARGET_FILE_DIR:sherpa-onnx-wasm-main-tts>/sherpa-onnx-wasm-main-tts.wasm"
    "$<TARGET_FILE_DIR:sherpa-onnx-wasm-main-tts>/sherpa-onnx-wasm-main-tts.data"
  DESTINATION
    bin/wasm/tts
)
