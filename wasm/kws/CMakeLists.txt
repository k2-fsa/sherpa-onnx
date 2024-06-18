if(NOT $ENV{SHERPA_ONNX_IS_USING_BUILD_WASM_SH})
    message(FATAL_ERROR "Please use ./build-wasm-simd-kws.sh to build for wasm KWS")
endif()

if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/assets/decoder-epoch-12-avg-2-chunk-16-left-64.onnx")
    message(WARNING "${CMAKE_CURRENT_SOURCE_DIR}/assets/decoder-epoch-12-avg-2-chunk-16-left-64.onnx does not exist")
    message(FATAL_ERROR "Please read ${CMAKE_CURRENT_SOURCE_DIR}/assets/README.md before you continue")
endif()

set(exported_functions
        AcceptWaveform
        CreateKeywordSpotter
        DestroyKeywordSpotter
        CreateKeywordStream
        DecodeKeywordStream
        GetKeywordResult
        DestroyKeywordResult
        IsKeywordStreamReady
        InputFinished
)
set(mangled_exported_functions)
foreach(x IN LISTS exported_functions)
    list(APPEND mangled_exported_functions "_${x}")
endforeach()

list(JOIN mangled_exported_functions "," all_exported_functions)

include_directories(${CMAKE_SOURCE_DIR})
set(MY_FLAGS "-s FORCE_FILESYSTEM=1 -s INITIAL_MEMORY=512MB -s ALLOW_MEMORY_GROWTH=1")
string(APPEND MY_FLAGS " -sSTACK_SIZE=10485760 ")
string(APPEND MY_FLAGS " -sEXPORTED_FUNCTIONS=[_CopyHeap,_malloc,_free,${all_exported_functions}] ")
string(APPEND MY_FLAGS "--preload-file ${CMAKE_CURRENT_SOURCE_DIR}/assets@. ")
string(APPEND MY_FLAGS " -sEXPORTED_RUNTIME_METHODS=['ccall','stringToUTF8','setValue','getValue','lengthBytesUTF8','UTF8ToString'] ")

message(STATUS "MY_FLAGS: ${MY_FLAGS}")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${MY_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${MY_FLAGS}")
set(CMAKE_EXECUTBLE_LINKER_FLAGS "${CMAKE_EXECUTBLE_LINKER_FLAGS} ${MY_FLAGS}")

add_executable(sherpa-onnx-wasm-kws-main sherpa-onnx-wasm-main-kws.cc)
target_link_libraries(sherpa-onnx-wasm-kws-main sherpa-onnx-c-api)
install(TARGETS sherpa-onnx-wasm-kws-main DESTINATION bin/wasm)

install(
        FILES
        "sherpa-onnx-kws.js"
        "app.js"
        "index.html"
        "$<TARGET_FILE_DIR:sherpa-onnx-wasm-kws-main>/sherpa-onnx-wasm-kws-main.js"
        "$<TARGET_FILE_DIR:sherpa-onnx-wasm-kws-main>/sherpa-onnx-wasm-kws-main.wasm"
        "$<TARGET_FILE_DIR:sherpa-onnx-wasm-kws-main>/sherpa-onnx-wasm-kws-main.data"
        DESTINATION
        bin/wasm
)
